import AVFoundation
import CoreImage
import UIKit

enum CameraError: LocalizedError {
    case noDevice
    case noInput
    case notAuthorized
    case configurationFailed(String)

    var errorDescription: String? {
        switch self {
        case .noDevice: return "找不到后置摄像头"
        case .noInput: return "无法添加相机输入"
        case .notAuthorized: return "相机权限未开启，请前往设置允许访问相机"
        case .configurationFailed(let m): return "相机初始化失败：\(m)"
        }
    }
}

// MARK: - 镜头

enum CameraLens: String, CaseIterable, Identifiable {
    case ultraWide
    case wide

    var id: String { rawValue }

    var shortLabel: String {
        switch self {
        case .ultraWide: return "0.5×"
        case .wide: return "1×"
        }
    }

    var title: String {
        switch self {
        case .ultraWide: return "超广角"
        case .wide: return "广角"
        }
    }

    var deviceType: AVCaptureDevice.DeviceType {
        switch self {
        case .ultraWide: return .builtInUltraWideCamera
        case .wide: return .builtInWideAngleCamera
        }
    }

    /// 送进拼接的中央比例。超广角边角分辨率骤降、残余畸变大，
    /// 只取中间一块做球面投影，接缝会干净很多（实测 0.92 与 0.88 张数相同、0.92 更清晰）。
    var usableCrop: Float {
        switch self {
        case .ultraWide: return 0.92
        case .wide: return 1.0
        }
    }

    /// 备用视场角（水平，传感器横向），读不到 format 时兜底
    var fallbackHorizontalFOV: Float {
        switch self {
        case .ultraWide: return 100
        case .wide: return 66
        }
    }
}

/// 竖屏下的相机内参（焦距已归一化）
struct CameraIntrinsics: Equatable {
    var tanHalfH: Float = 0.47     // 竖屏水平半视场角正切（全画幅）
    var tanHalfV: Float = 0.63     // 竖屏垂直半视场角正切（全画幅）
    var crop: Float = 1.0          // 参与拼接的中央比例

    var tanHalfHEffective: Float { tanHalfH * crop }
    var tanHalfVEffective: Float { tanHalfV * crop }
    var hfovDegrees: Float { 2 * atan(tanHalfHEffective) * 180 / .pi }
    var vfovDegrees: Float { 2 * atan(tanHalfVEffective) * 180 / .pi }
}

struct CameraFrame {
    let buffer: CVPixelBuffer
    let timestamp: TimeInterval    // 与 CACurrentMediaTime / CMDeviceMotion 同一时钟
    let luma: Float
}

/// 采集相机视频流（而不是 PhotoOutput）：每一帧都是已知朝向、已知时间戳的 CVPixelBuffer，
/// 与陀螺仪姿态可以严格对齐。
final class CameraController: NSObject {

    let session = AVCaptureSession()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "pano.camera.session")
    private let bufferQueue = DispatchQueue(label: "pano.camera.buffer", qos: .userInitiated)
    private let ciContext = CIContext(options: [.cacheIntermediates: false])

    private let lock = NSLock()
    private var latest: (buffer: CVPixelBuffer, timestamp: TimeInterval)?

    private(set) var intrinsics = CameraIntrinsics()
    private(set) var activeLens: CameraLens = .wide
    private(set) var supportsUltraWide = false
    private(set) var imageSize: CGSize = .zero
    private(set) var isConfigured = false
    private(set) var isRunning = false

    /// 竖屏 90° 旋转后的缓冲区尺寸
    static let portraitRotationAngle: CGFloat = 90

    // MARK: - 生命周期

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    private static func device(for lens: CameraLens) -> AVCaptureDevice? {
        AVCaptureDevice.default(lens.deviceType, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(for: .video)
    }

    func configure() async throws {
        guard await CameraController.requestAccess() else { throw CameraError.notAuthorized }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    try self.configureLocked()
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    private func configureLocked() throws {
        guard !isConfigured else { return }
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .photo

        supportsUltraWide = AVCaptureDevice.default(.builtInUltraWideCamera,
                                                   for: .video,
                                                   position: .back) != nil
        // 默认用超广角：一次能覆盖 90°×106°，整圈只要十来张，拍摄速度快一个量级
        let lens: CameraLens = supportsUltraWide ? .ultraWide : .wide

        guard let device = CameraController.device(for: lens) else {
            throw CameraError.noDevice
        }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.noInput }
        session.addInput(input)
        activeLens = lens

        applyBestFormat(on: device)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: bufferQueue)
        guard session.canAddOutput(videoOutput) else {
            throw CameraError.configurationFailed("无法添加视频输出")
        }
        session.addOutput(videoOutput)

        if let conn = videoOutput.connection(with: .video),
           conn.isVideoRotationAngleSupported(CameraController.portraitRotationAngle) {
            conn.videoRotationAngle = CameraController.portraitRotationAngle
        }

        configureIntrinsics(on: device)
        isConfigured = true
    }

    /// 切换镜头（0.5× ⇄ 1×）。会话运行中换输入，预览层会自动跟上。
    func setLens(_ lens: CameraLens) async throws {
        if lens == .ultraWide && !supportsUltraWide { return }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    try self.applyLensLocked(lens)
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    private func applyLensLocked(_ lens: CameraLens) throws {
        guard isConfigured, lens != activeLens else { return }
        guard let device = CameraController.device(for: lens) else { throw CameraError.noDevice }
        let newInput = try AVCaptureDeviceInput(device: device)

        session.beginConfiguration()
        for input in session.inputs { session.removeInput(input) }
        guard session.canAddInput(newInput) else {
            session.commitConfiguration()
            throw CameraError.noInput
        }
        session.addInput(newInput)
        activeLens = lens
        applyBestFormat(on: device)
        session.commitConfiguration()

        if let conn = videoOutput.connection(with: .video),
           conn.isVideoRotationAngleSupported(CameraController.portraitRotationAngle) {
            conn.videoRotationAngle = CameraController.portraitRotationAngle
        }

        configureIntrinsics(on: device)
        lock.lock()
        latest = nil
        lock.unlock()
    }

    // MARK: - 格式与内参

    /// 优先 4:3：竖屏下竖直方向的视场更宽，同样张数覆盖更多球面。
    private func applyBestFormat(on device: AVCaptureDevice, maxEdge: Int = 2400) {
        func pick(limitEdge: Int?) -> AVCaptureDevice.Format? {
            var best: AVCaptureDevice.Format?
            var bestPixels = 0
            for f in device.formats {
                guard !f.videoSupportedFrameRateRanges.isEmpty else { continue }
                let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                let w = Int(d.width), h = Int(d.height)
                guard w > 0, h > 0 else { continue }
                let ratio = Double(max(w, h)) / Double(min(w, h))
                guard abs(ratio - 4.0 / 3.0) < 0.04 else { continue }
                if let limitEdge, max(w, h) > limitEdge { continue }
                let px = w * h
                if px > bestPixels {
                    bestPixels = px
                    best = f
                }
            }
            return best
        }

        var format = pick(limitEdge: maxEdge) ?? pick(limitEdge: nil)

        if format == nil {
            var best: AVCaptureDevice.Format?
            var bestPixels = 0
            for f in device.formats {
                guard !f.videoSupportedFrameRateRanges.isEmpty else { continue }
                let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                let px = Int(d.width) * Int(d.height)
                if px > bestPixels {
                    bestPixels = px
                    best = f
                }
            }
            format = best
        }
        guard let format else { return }

        do {
            try device.lockForConfiguration()
            // 开启镜头几何畸变校正：超广角的桶形畸变会被拍平，
            // 否则边缘的直线弯成弧，球面投影后接缝一定会错开。
            if device.isGeometricDistortionCorrectionSupported {
                device.isGeometricDistortionCorrectionEnabled = true
            }
            device.activeFormat = format
            let duration = CMTime(value: 1, timescale: 30)
            if format.videoSupportedFrameRateRanges.contains(where: { $0.minFrameRate <= 30 && $0.maxFrameRate >= 30 }) {
                device.activeVideoMinFrameDuration = duration
                device.activeVideoMaxFrameDuration = duration
            }
            device.videoZoomFactor = 1.0
            if device.isLowLightBoostSupported { device.automaticallyEnablesLowLightBoostWhenAvailable = true }
            device.unlockForConfiguration()
        } catch {
            // 保持默认格式
        }
    }

    private func configureIntrinsics(on device: AVCaptureDevice) {
        let format = device.activeFormat
        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)

        // videoFieldOfView 是「传感器横向（长边）」的水平视场角；
        // 开了畸变校正后要以校正后的视场角为准，否则焦距会算偏，接缝会整体错位。
        var deg = format.videoFieldOfView
        if device.isGeometricDistortionCorrectionEnabled {
            let corrected = format.geometricDistortionCorrectedVideoFieldOfView
            if corrected > 5 { deg = corrected }
        }

        let longEdge = Float(max(dims.width, dims.height))
        let shortEdge = Float(min(dims.width, dims.height))
        let fallback = activeLens.fallbackHorizontalFOV
        let hfovDeg: Float = (deg > 12 && deg < 150) ? Float(deg) : fallback
        let focal = (longEdge / 2) / tan(hfovDeg * .pi / 360)

        // 竖屏：长边在竖直方向
        intrinsics = CameraIntrinsics(tanHalfH: (shortEdge / 2) / focal,
                                      tanHalfV: (longEdge / 2) / focal,
                                      crop: activeLens.usableCrop)
        imageSize = CGSize(width: CGFloat(shortEdge), height: CGFloat(longEdge))
    }

    // MARK: - 运行

    func start() {
        sessionQueue.async {
            guard self.isConfigured, !self.session.isRunning else { return }
            self.session.startRunning()
            self.isRunning = true

            // 看门狗：2.5s 内没有出图（有些机型不支持高分辨率视频流）就降档
            self.sessionQueue.asyncAfter(deadline: .now() + 2.5) { [weak self] in
                guard let self, self.isRunning, !self.hasReceivedFrame else { return }
                self.downgradeFormat()
            }
        }
    }

    private var hasReceivedFrame: Bool {
        lock.lock()
        defer { lock.unlock() }
        return latest != nil
    }

    private func downgradeFormat() {
        guard let device = session.inputs.compactMap({ ($0 as? AVCaptureDeviceInput)?.device }).first else { return }
        session.beginConfiguration()
        applyBestFormat(on: device, maxEdge: 2000)
        session.commitConfiguration()
        configureIntrinsics(on: device)
    }

    func stop() {
        sessionQueue.async {
            guard self.session.isRunning else { return }
            self.session.stopRunning()
            self.isRunning = false
            self.lock.lock()
            self.latest = nil
            self.lock.unlock()
        }
    }

    // MARK: - 取帧

    /// 取出最新一帧（保留引用，避免被缓冲池回收）
    func grabFrame() -> CameraFrame? {
        lock.lock()
        let frame = latest
        lock.unlock()
        guard let frame else { return nil }
        return CameraFrame(buffer: frame.buffer,
                           timestamp: frame.timestamp,
                           luma: CameraController.meanLuma(of: frame.buffer))
    }

    /// 把 CVPixelBuffer 编码成 JPEG（后台执行）
    func encodeJPEG(_ buffer: CVPixelBuffer, quality: CGFloat = 0.92) -> Data? {
        let ci = CIImage(cvPixelBuffer: buffer)
        guard let cg = ciContext.createCGImage(ci, from: ci.extent) else { return nil }
        return UIImage(cgImage: cg, scale: 1, orientation: .up).jpegData(compressionQuality: quality)
    }

    static func meanLuma(of buffer: CVPixelBuffer) -> Float {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard CVPixelBufferGetPlaneCount(buffer) > 0,
              let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return 0.5 }
        let w = CVPixelBufferGetWidthOfPlane(buffer, 0)
        let h = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let bpr = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        var sum: Double = 0
        var count = 0
        let step = 24
        var y = 0
        while y < h {
            var x = 0
            while x < w {
                sum += Double(ptr[y * bpr + x])
                count += 1
                x += step
            }
            y += step
        }
        return count > 0 ? Float(sum / Double(count) / 255.0) : 0.5
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        lock.lock()
        latest = (pb, pts)
        lock.unlock()
    }
}
