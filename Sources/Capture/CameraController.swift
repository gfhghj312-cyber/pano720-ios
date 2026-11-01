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

/// 采集相机视频流（而不是 PhotoOutput）：每一帧都是已知朝向的 CVPixelBuffer，
/// 与陀螺仪姿态可以严格对齐，拼接质量更稳定。
final class CameraController: NSObject {

    let session = AVCaptureSession()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "pano.camera.session")
    private let bufferQueue = DispatchQueue(label: "pano.camera.buffer", qos: .userInitiated)
    private let ciContext = CIContext(options: [.cacheIntermediates: false])

    private let lock = NSLock()
    private var latestBuffer: CVPixelBuffer?

    private(set) var imageSize: CGSize = .zero
    private(set) var focalPixels: Float = 3000
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

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                ?? AVCaptureDevice.default(for: .video) else {
            throw CameraError.noDevice
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.noInput }
        session.addInput(input)

        applyBestFormat(on: device)
        computeIntrinsics(for: device)

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

        // 旋转 90° 后缓冲区的宽高互换
        imageSize = CGSize(width: imageSize.height, height: imageSize.width)
        isConfigured = true
    }

    /// 优先选择 4:3 最大尺寸的视频格式：竖屏下水平 FOV 更宽，拍摄张数更少
    private func applyBestFormat(on device: AVCaptureDevice) {
        var best: AVCaptureDevice.Format?
        var bestPixels = 0
        for f in device.formats {
            guard !f.videoSupportedFrameRateRanges.isEmpty else { continue }
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            let w = Int(d.width), h = Int(d.height)
            guard w > 0, h > 0 else { continue }
            let ratio = Double(max(w, h)) / Double(min(w, h))
            guard abs(ratio - 4.0 / 3.0) < 0.04 else { continue }
            let px = w * h
            if px > bestPixels {
                bestPixels = px
                best = f
            }
        }
        if best == nil {
            for f in device.formats {
                guard !f.videoSupportedFrameRateRanges.isEmpty else { continue }
                let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                let px = Int(d.width) * Int(d.height)
                if px > bestPixels {
                    bestPixels = px
                    best = f
                }
            }
        }
        guard let format = best else { return }

        do {
            try device.lockForConfiguration()
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

    private func computeIntrinsics(for device: AVCaptureDevice) {
        let format = device.activeFormat
        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        imageSize = CGSize(width: CGFloat(dims.width), height: CGFloat(dims.height))

        let longEdge = Float(max(dims.width, dims.height))

        // videoFieldOfView 是「长边方向」的水平视场角（传感器原生横向）
        let deg = Float(format.videoFieldOfView)
        let hfov: Float = (deg > 10 && deg < 140) ? deg * Float.pi / 180 : 66 * Float.pi / 180
        focalPixels = (longEdge / 2) / tan(hfov / 2)
    }

    // MARK: - 运行

    func start() {
        sessionQueue.async {
            guard self.isConfigured, !self.session.isRunning else { return }
            self.session.startRunning()
            self.isRunning = true
        }
    }

    func stop() {
        sessionQueue.async {
            guard self.session.isRunning else { return }
            self.session.stopRunning()
            self.isRunning = false
            self.lock.lock()
            self.latestBuffer = nil
            self.lock.unlock()
        }
    }

    // MARK: - 取帧

    /// 取出最新一帧（保留引用，避免被缓冲池回收）
    func grabFrame() -> (buffer: CVPixelBuffer, luma: Float)? {
        lock.lock()
        let buffer = latestBuffer
        lock.unlock()
        guard let buffer else { return nil }
        return (buffer, CameraController.meanLuma(of: buffer))
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
        lock.lock()
        latestBuffer = pb
        lock.unlock()
    }
}
