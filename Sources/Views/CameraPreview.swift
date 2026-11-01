import SwiftUI
import AVFoundation

/// AVCaptureVideoPreviewLayer 容器。竖屏固定 90° 旋转，与采集缓冲区保持一致。
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
        uiView.applyRotation()
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        applyRotation()
    }

    func applyRotation() {
        guard let connection = previewLayer.connection else { return }
        let angle = CameraController.portraitRotationAngle
        if connection.isVideoRotationAngleSupported(angle), connection.videoRotationAngle != angle {
            connection.videoRotationAngle = angle
        }
    }
}
