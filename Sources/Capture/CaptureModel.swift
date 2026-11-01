import Foundation
import UIKit
import QuartzCore
import Observation
import simd

/// 一次拍摄会话的全部状态机：网格生成、对齐判定、自动快门、拼接调度
@Observable
final class CaptureModel {

    struct Shot: Identifiable {
        let id = UUID()
        let fileName: String
        let rotation: simd_float3x3
        let luma: Float
        let nodeID: Int?
        let thumbnail: UIImage?
    }

    enum Phase: Equatable {
        case preparing      // 相机启动中
        case ready          // 可拍摄
        case capturing      // 正在写盘
        case stitching      // 拼接中
        case finished
        case failed(String)
    }

    // MARK: - 输出状态（供视图观察）

    private(set) var nodes: [CaptureNode] = []
    private(set) var shots: [Shot] = []
    private(set) var phase: Phase = .preparing
    private(set) var currentRotation = matrix_identity_float3x3
    private(set) var currentForward = Vec3(0, 1, 0)
    private(set) var targetID: Int?
    private(set) var alignmentDeg: Float = 180
    private(set) var isAligned = false
    private(set) var isSteady = false
    private(set) var isAutoCaptureEnabled = true
    private(set) var flashTick = 0
    private(set) var lastThumbnail: UIImage?
    private(set) var stitchProgress: Double = 0
    private(set) var imageAspect: CGFloat = 0.75
    var tanHalfH: Float = 0.47
    var tanHalfV: Float = 0.63

    let camera = CameraController()
    private let motion = MotionService()
    private let encoderQueue = DispatchQueue(label: "pano.encode", qos: .userInitiated)

    private var history: [(TimeInterval, Vec3)] = []
    private var lastShotTime: TimeInterval = 0
    private var sessionDir: URL?
    private let haptic = UIImpactFeedbackGenerator(style: .rigid)

    // MARK: - 派生量

    var capturedCount: Int { nodes.filter(\.captured).count }

    var requiredCount: Int { max(1, nodes.filter { !$0.isOptional }.count) }

    var progress: Double {
        guard !nodes.isEmpty else { return 0 }
        return Double(capturedCount) / Double(nodes.count)
    }

    var isCompleteEnough: Bool {
        capturedCount >= Int(Double(requiredCount) * 0.85)
    }

    var targetNode: CaptureNode? {
        guard let targetID else { return nil }
        return nodes.first { $0.id == targetID }
    }

    /// 相对当前朝向，目标在左边还是右边（用于箭头提示）
    var targetOffsetYaw: Float {
        guard let node = targetNode else { return 0 }
        return wrapAngle(yawOf(node.direction) - yawOf(currentForward))
    }

    var targetOffsetPitch: Float {
        guard let node = targetNode else { return 0 }
        return pitchOf(node.direction) - pitchOf(currentForward)
    }

    var hintText: String {
        switch phase {
        case .preparing: return "正在启动相机…"
        case .capturing: return "已拍摄，请稍等…"
        case .stitching: return "正在拼接全景…"
        case .failed(let m): return m
        case .finished: return "完成"
        case .ready:
            if targetNode == nil { return "全部拍完，点「拼接」生成全景" }
            if !isSteady { return "端稳手机，慢慢移动" }
            if isAligned { return "保持不动，正在自动拍摄" }
            let yaw = targetOffsetYaw
            let pitch = targetOffsetPitch
            if abs(yaw) > 0.10 {
                return yaw > 0 ? "向右转 \(Int(abs(rad2deg(yaw))))°" : "向左转 \(Int(abs(rad2deg(yaw))))°"
            }
            return pitch > 0 ? "稍微抬高手机" : "稍微压低手机"
        }
    }

    // MARK: - 生命周期

    func start() {
        guard case .preparing = phase else { return }
        Task { @MainActor in
            do {
                try await camera.configure()
                camera.start()
                let size = camera.imageSize
                if size.width > 0 {
                    imageAspect = size.width / size.height
                    tanHalfH = Float(size.width) / 2 / camera.focalPixels
                    tanHalfV = Float(size.height) / 2 / camera.focalPixels
                }
                prepareSessionDirectory()
                motion.onSample = { [weak self] sample in
                    self?.handle(sample)
                }
                motion.start()
                haptic.prepare()
                phase = .ready
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    func stop() {
        motion.stop()
        motion.onSample = nil
        camera.stop()
        if case .stitching = phase { return }
        phase = .preparing
    }

    func toggleAutoCapture() {
        isAutoCaptureEnabled.toggle()
    }

    // MARK: - 姿态处理

    private func handle(_ sample: MotionSample) {
        currentRotation = sample.rotation
        currentForward = sample.forward

        if nodes.isEmpty {
            // 首次姿态决定网格的起始方位，让 0 号机位正对用户前方
            nodes = PanoramaGrid.nodes(tanHalfH: tanHalfH,
                                       tanHalfV: tanHalfV,
                                       baseYaw: yawOf(sample.forward))
        }

        history.append((sample.timestamp, sample.forward))
        while let first = history.first, sample.timestamp - first.0 > 0.40 {
            history.removeFirst()
        }

        // 0.4 秒内光轴抖动小于 0.9° 视为稳定
        var steady = history.count >= 6
        if steady {
            for item in history where angleBetween(item.1, sample.forward) > 0.016 {
                steady = false
                break
            }
        }
        isSteady = steady

        guard case .ready = phase else { return }

        // 目标机位 = 最近的未拍摄节点
        guard let nextTarget = nearestPendingNode() else {
            targetID = nil
            isAligned = false
            alignmentDeg = 0
            return
        }
        targetID = nextTarget.id
        let err = angleBetween(sample.forward, nextTarget.direction)
        alignmentDeg = rad2deg(err)
        isAligned = alignmentDeg < 2.2

        if isAutoCaptureEnabled, isAligned, isSteady, sample.timestamp - lastShotTime > 0.65 {
            triggerCapture(nodeID: nextTarget.id)
        }
    }

    private func nearestPendingNode() -> CaptureNode? {
        var best: CaptureNode?
        var bestAngle: Float = .greatestFiniteMagnitude
        for node in nodes where !node.captured {
            let a = angleBetween(currentForward, node.direction)
            if a < bestAngle {
                bestAngle = a
                best = node
            }
        }
        return best
    }

    // MARK: - 拍摄

    /// 手动快门：对准了就按当前机位记录，没对准也允许拍（像素仍然有用）
    func manualCapture() {
        guard case .ready = phase else { return }
        let nodeID: Int? = (alignmentDeg < 12) ? targetID : nil
        triggerCapture(nodeID: nodeID)
    }

    private func triggerCapture(nodeID: Int?) {
        guard case .ready = phase else { return }
        guard let frame = camera.grabFrame() else { return }

        phase = .capturing
        lastShotTime = CACurrentMediaTime()

        let rotation = currentRotation
        let index = shots.count
        let dir = sessionDir
        let buffer = frame.buffer
        let luma = frame.luma

        flashTick &+= 1
        haptic.impactOccurred()

        encoderQueue.async { [weak self] in
            guard let self else { return }
            let data = self.camera.encodeJPEG(buffer)
            var thumb: UIImage?
            var name: String?
            if let data, let dir {
                let fileName = String(format: "shot-%03d.jpg", index)
                let url = dir.appendingPathComponent(fileName)
                if (try? data.write(to: url)) != nil {
                    name = fileName
                    if let image = UIImage(data: data) {
                        thumb = CaptureModel.smallThumbnail(from: image)
                    }
                }
            }
            DispatchQueue.main.async {
                if let name {
                    self.shots.append(Shot(fileName: name, rotation: rotation, luma: luma,
                                           nodeID: nodeID, thumbnail: thumb))
                    if let nodeID, let i = self.nodes.firstIndex(where: { $0.id == nodeID }) {
                        self.nodes[i].captured = true
                    }
                    self.lastThumbnail = thumb
                }
                self.phase = .ready
            }
        }
    }

    func undoLast() {
        guard case .ready = phase, let last = shots.popLast() else { return }
        if let nodeID = last.nodeID, let i = nodes.firstIndex(where: { $0.id == nodeID }) {
            nodes[i].captured = false
        }
        if let dir = sessionDir {
            try? FileManager.default.removeItem(at: dir.appendingPathComponent(last.fileName))
        }
        lastThumbnail = shots.last?.thumbnail
        lastShotTime = 0
    }

    // MARK: - 拼接

    @MainActor
    func stitch() async throws -> StitchedPanorama {
        guard let dir = sessionDir else { throw StitchError.noInput }
        phase = .stitching
        stitchProgress = 0

        let inputs = shots.map { shot in
            StitchInput(url: dir.appendingPathComponent(shot.fileName),
                        rotation: shot.rotation,
                        tanHalfH: tanHalfH,
                        tanHalfV: tanHalfV,
                        luma: shot.luma)
        }

        do {
            let result = try await StitchService.stitch(inputs: inputs, outWidth: 4096) { [weak self] p in
                DispatchQueue.main.async { self?.stitchProgress = p }
            }
            phase = .finished
            return result
        } catch {
            phase = .failed(error.localizedDescription)
            throw error
        }
    }

    func cleanupWorkingFiles() {
        guard let dir = sessionDir else { return }
        try? FileManager.default.removeItem(at: dir)
        sessionDir = nil
    }

    // MARK: - 私有

    private func prepareSessionDirectory() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("pano-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        sessionDir = dir
    }

    private static func smallThumbnail(from image: UIImage, maxDim: CGFloat = 240) -> UIImage? {
        let w = image.size.width
        let h = image.size.height
        guard w > 0, h > 0 else { return nil }
        let scale = min(1, maxDim / max(w, h))
        let size = CGSize(width: w * scale, height: h * scale)
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
    }
}
