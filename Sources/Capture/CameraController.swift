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
    /// 第一帧到达后按**实际缓冲区尺寸**修正的内参。
    /// 不能信 `videoRotationAngle`：Apple 文档只说「设置角度不一定旋转像素」，
    /// 支持该角度也不代表会生效（前置摄像头实测会被静默忽略）。
    /// 一旦没生效，缓冲区仍是横向的，而按竖屏假设算出的 tanH/tanV 正好宽高颠倒，
    /// 每帧会被压扁 1.78 倍 —— 拼出来就是彻底的错乱。
    var onIntrinsicsChange: ((CameraIntrinsics) -> Void)?
    private var lastBufferSize: CGSize = .zero
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

        // inputPriority：由我们自己指定 activeFormat。
        // 用 .photo/.high 这类预设时，commitConfiguration 会把 activeFormat 重置成预设选的格式，
        // 焦距标定跟着变 —— 这是接缝对不上的一个隐性来源。
        session.sessionPreset = .inputPriority

        supportsUltraWide = AVCaptureDevice.default(.builtInUltraWideCamera,
                                                   for: .video,
                                                   position: .back) != nil
        // 默认 1×：同样张数下角分辨率高 2.4 倍，成片明显更清晰
        let lens: CameraLens = .wide

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

    /// 切换镜头（0.5× ⇄ 1×）。
    ///
    /// 运行中的会话直接换 input，在某些机型上会静默失败：会话还在跑，但预览和出图都停在旧设备上，
    /// 用户看到的就是「点了没反应」。所以先停会话、换输入、设格式，再重启；失败就回滚到旧镜头。
    func setLens(_ lens: CameraLens) async throws {
        if lens == .ultraWide && !supportsUltraWide { throw CameraError.noDevice }
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
        guard isConfigured else { throw CameraError.configurationFailed("相机未就绪") }
        guard lens != activeLens else { return }
        let previous = activeLens

        guard let device = CameraController.device(for: lens) else { throw CameraError.noDevice }
        let newInput = try AVCaptureDeviceInput(device: device)

        let wasRunning = session.isRunning
        if wasRunning { session.stopRunning() }

        session.beginConfiguration()
        let oldInputs = session.inputs
        for input in oldInputs { session.removeInput(input) }
        guard session.canAddInput(newInput) else {
            // 回滚
            for input in oldInputs { session.addInput(input) }
            session.commitConfiguration()
            if wasRunning { session.startRunning() }
            throw CameraError.noInput
        }
        session.addInput(newInput)
        session.commitConfiguration()

        // 格式在 commit 之后设：inputPriority 下不会被预设覆盖
        applyBestFormat(on: device)
        if let conn = videoOutput.connection(with: .video),
           conn.isVideoRotationAngleSupported(CameraController.portraitRotationAngle) {
            conn.videoRotationAngle = CameraController.portraitRotationAngle
        }

        activeLens = lens
        configureIntrinsics(on: device)

        lock.lock()
        latest = nil
        // 换镜头后视场角变了，但缓冲区尺寸可能完全相同，
        // 清掉尺寸缓存才能强制下一帧重算内参
        lastBufferSize = .zero
        lock.unlock()

        if wasRunning { session.startRunning() }

        // 出图看门狗：换完镜头 1.6s 还没新帧，说明这台机器不支持这种换法，退回旧镜头
        if wasRunning, !waitForFreshFrame(timeout: 1.6) {
            session.stopRunning()
            session.beginConfiguration()
            for input in session.inputs { session.removeInput(input) }
            if let old = oldInputs.first, session.canAddInput(old) {
                session.addInput(old)
            } else if let fallback = CameraController.device(for: previous),
                      let input = try? AVCaptureDeviceInput(device: fallback),
                      session.canAddInput(input) {
                session.addInput(input)
            }
            session.commitConfiguration()
            if let d = session.inputs.compactMap({ ($0 as? AVCaptureDeviceInput)?.device }).first {
                applyBestFormat(on: d)
                configureIntrinsics(on: d)
            }
            activeLens = previous
            session.startRunning()
            throw CameraError.configurationFailed("这台设备无法切换到\(lens.title)，已保持\(previous.title)")
        }
    }

    /// 阻塞等待一帧新画面（在 sessionQueue 上调用）
    private func waitForFreshFrame(timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            lock.lock()
            let has = latest != nil
            lock.unlock()
            if has { return true }
            Thread.sleep(forTimeInterval: 0.05)
        }
        return false
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
            device.activeFormat = format
            // 畸变校正要在 activeFormat 之后打开：它是按格式生效的，
            // 先开再改格式在部分机型上会被重置掉。超广角的桶形畸变不校正，
            // 边缘的直线就是弯的，球面投影后接缝必然错开。
            if device.isGeometricDistortionCorrectionSupported {
                device.isGeometricDistortionCorrectionEnabled = true
            }
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
        // 首帧之前的猜测值：按「竖屏」假设，长边在竖直方向。
        // 第一帧到达后会被 intrinsicsFor(size:device:) 用真实缓冲区尺寸覆盖。
        let dims = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        let guess = CGSize(width: CGFloat(min(dims.width, dims.height)),
                           height: CGFloat(max(dims.width, dims.height)))
        intrinsics = intrinsicsFor(size: guess, device: device)
        imageSize = guess
    }

    /// 用真实缓冲区尺寸算内参。
    ///
    /// 关键点：`videoFieldOfView`（以及畸变校正后的 corrected 版本）描述的是
    /// **传感器长边**方向的视场角。长边在成片里是横是竖，只有缓冲区尺寸知道，
    /// 所以这里一律用 `max(w,h)` 定位长边，再按宽高比例分配水平/垂直。
    /// 这样无论连接是否真的旋转了像素，结果都是自洽的。
    private func intrinsicsFor(size: CGSize, device: AVCaptureDevice) -> CameraIntrinsics {
        let format = device.activeFormat
        var deg = format.videoFieldOfView
        // 畸变校正会裁切并重采样边缘，视场角随之变化；不跟着改，焦距会整体偏。
        if device.isGeometricDistortionCorrectionEnabled {
            let corrected = format.geometricDistortionCorrectedVideoFieldOfView
            if corrected > 5 { deg = corrected }
        }

        let fallback = activeLens.fallbackHorizontalFOV
        let fovDeg: Float = (deg > 12 && deg < 150) ? Float(deg) : fallback

        let bw = Float(size.width)
        let bh = Float(size.height)
        guard bw > 1, bh > 1 else { return intrinsics }

        let longEdge = max(bw, bh)
        let focal = (longEdge / 2) / tan(fovDeg * .pi / 360)
        guard focal > 1 else { return intrinsics }

        return CameraIntrinsics(tanHalfH: (bw / 2) / focal,
                                tanHalfV: (bh / 2) / focal,
                                crop: activeLens.usableCrop)
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
    ///
    /// 这里会强制把画面统一成「竖屏」输出。原因是 `videoRotationAngle` 只是请求，
    /// Apple 文档明确写了「设置角度不一定旋转像素」，而且连接支持该角度也不代表会生效
    /// （前置摄像头实测会被静默忽略）。一旦没生效，缓冲区仍是横向的，
    /// 此时图像的水平轴对应设备的竖直轴，球面投影会整体错 90° —— 拼出来必然错乱。
    func encodeJPEG(_ buffer: CVPixelBuffer, quality: CGFloat = 0.92) -> Data? {
        var ci = CIImage(cvPixelBuffer: buffer)
        if ci.extent.height > 0, ci.extent.width > ci.extent.height {
            ci = ci.oriented(.right)
        }
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

        let w = CGFloat(CVPixelBufferGetWidth(pb))
        let h = CGFloat(CVPixelBufferGetHeight(pb))
        let size = CGSize(width: w, height: h)

        lock.lock()
        let sizeChanged = (size != lastBufferSize)
        if sizeChanged { lastBufferSize = size }
        latest = (pb, pts)
        lock.unlock()

        // 第一帧 / 换镜头后：用真实尺寸校正内参，并通知上层重建机位网格
        if sizeChanged, w > 1, h > 1 {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                guard let device = self.session.inputs
                    .compactMap({ ($0 as? AVCaptureDeviceInput)?.device }).first else { return }
                // 编码阶段会把横向缓冲区旋成竖屏，内参必须按「最终竖屏尺寸」算，
                // 否则 tanH/tanV 会和实际画幅宽高对不上
                let oriented = w > h ? CGSize(width: h, height: w) : size
                let updated = self.intrinsicsFor(size: oriented, device: device)
                self.imageSize = oriented
                if updated != self.intrinsics {
                    self.intrinsics = updated
                    self.onIntrinsicsChange?(updated)
                }
            }
        }
    }
}
