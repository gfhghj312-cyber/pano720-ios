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
        let rotation: simd_float3x3     // 相机 -> 世界（按该帧时间戳插值得到）
        let intrinsics: CameraIntrinsics
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
    private(set) var lens: CameraLens = .wide
    private(set) var isSwitchingLens = false

    let camera = CameraController()
    private let motion = MotionService()
    private let encoderQueue = DispatchQueue(label: "pano.encode", qos: .userInitiated)

    private var history: [(TimeInterval, Vec3)] = []
    private var lastShotTime: TimeInterval = 0
    private var sessionDir: URL?
    private let haptic = UIImpactFeedbackGenerator(style: .rigid)

    // MARK: - 派生量

    /// 取景器/引导用：全画幅半视场角正切（不含裁切）
    var tanHalfH: Float { camera.intrinsics.tanHalfH }
    var tanHalfV: Float { camera.intrinsics.tanHalfV }
    var supportsUltraWide: Bool { camera.supportsUltraWide }
    var fovText: String {
        let i = camera.intrinsics
        return "\(Int(i.hfovDegrees.rounded()))°×\(Int(i.vfovDegrees.rounded()))°"
    }

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
            if isSwitchingLens { return "正在切换镜头…" }
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
                lens = camera.activeLens
                applyCameraGeometry()
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

    /// 切换 0.5× / 1×：视场角变了，机位网格必须重算，已拍的帧也要清掉
    /// （同一张全景里混用两种焦距会让重叠区域的尺度对不上）
    func switchLens(_ newLens: CameraLens) {
        guard newLens != lens, !isSwitchingLens else { return }
        guard newLens != .ultraWide || camera.supportsUltraWide else { return }
        isSwitchingLens = true
        clearShots()
        Task { @MainActor in
            do {
                try await camera.setLens(newLens)
                lens = newLens
                applyCameraGeometry()
                rebuildGrid()
                isSwitchingLens = false
                phase = .ready
            } catch {
                isSwitchingLens = false
                phase = .failed(error.localizedDescription)
            }
        }
    }

    private func applyCameraGeometry() {
        let size = camera.imageSize
        if size.width > 0, size.height > 0 {
            imageAspect = size.width / size.height
        }
    }

    // MARK: - 姿态处理

    private func handle(_ sample: MotionSample) {
        currentRotation = sample.rotation
        currentForward = sample.forward

        if nodes.isEmpty {
            rebuildGrid(forward: sample.forward)
        }

        history.append((sample.timestamp, sample.forward))
        while let first = history.first, sample.timestamp - first.0 > 0.40 {
            history.removeFirst()
        }

        // 0.4 秒内光轴抖动小于约 0.9° 视为稳定
        var steady = history.count >= 6
        if steady {
            for item in history where angleBetween(item.1, sample.forward) > 0.016 {
                steady = false
                break
            }
        }
        isSteady = steady

        guard case .ready = phase, !isSwitchingLens else { return }

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

        if isAutoCaptureEnabled, isAligned, isSteady, sample.timestamp - lastShotTime > 0.55 {
            triggerCapture(nodeID: nextTarget.id)
        }
    }

    private func rebuildGrid(forward: Vec3? = nil) {
        let base = forward ?? currentForward
        let intrinsics = camera.intrinsics
        nodes = PanoramaGrid.nodes(tanHalfH: intrinsics.tanHalfHEffective,
                                   tanHalfV: intrinsics.tanHalfVEffective,
                                   overlap: 0.16,
                                   baseDirection: base)
        targetID = nil
        alignmentDeg = 180
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
        guard case .ready = phase, !isSwitchingLens else { return }
        let nodeID: Int? = (alignmentDeg < 12) ? targetID : nil
        triggerCapture(nodeID: nodeID)
    }

    private func triggerCapture(nodeID: Int?) {
        guard case .ready = phase, !isSwitchingLens else { return }
        guard let frame = camera.grabFrame() else { return }

        phase = .capturing
        lastShotTime = CACurrentMediaTime()

        // 按「这一帧的出图时间」取姿态：视频流有约 1~2 帧的缓冲，
        // 直接拿当前陀螺仪读数会在转动的瞬间差出 1~3°，接缝就错开了。
        // 若两个时间轴不同源（偏差过大），退回当前姿态，绝不拿旧姿态顶替。
        let now = CACurrentMediaTime()
        var rotation = currentRotation
        if abs(now - frame.timestamp) < 0.25,
           let matched = motion.rotation(at: frame.timestamp) {
            rotation = matched
        }
        let intrinsics = camera.intrinsics
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
                    self.shots.append(Shot(fileName: name, rotation: rotation,
                                           intrinsics: intrinsics, luma: luma,
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

    private func clearShots() {
        shots.removeAll()
        lastThumbnail = nil
        lastShotTime = 0
        nodes = []
        targetID = nil
        if let dir = sessionDir {
            try? FileManager.default.removeItem(at: dir)
        }
        prepareSessionDirectory()
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
                        tanHalfH: shot.intrinsics.tanHalfH,
                        tanHalfV: shot.intrinsics.tanHalfV,
                        crop: shot.intrinsics.crop,
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
