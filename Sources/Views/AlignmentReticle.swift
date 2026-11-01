import SwiftUI
import simd

/// 取景器上的对准准星 + 方向指引
struct AlignmentReticle: View {
    var model: CaptureModel

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let aligned = model.isAligned && model.isSteady
            let color: Color = aligned ? Palette.mint : (model.isSteady ? .white : Palette.amber)
            let target = projected(model: model, size: size)

            ZStack {
                // 中心十字准星
                ReticleShape(aligned: aligned)
                    .stroke(color.opacity(aligned ? 0.95 : 0.55),
                            style: StrokeStyle(lineWidth: aligned ? 2.6 : 1.6, lineCap: .round))
                    .frame(width: reticleSize(aligned: aligned), height: reticleSize(aligned: aligned))
                    .scaleEffect(aligned ? 0.92 : 1)
                    .animation(.spring(response: 0.35, dampingFraction: 0.7), value: aligned)
                    .position(x: size.width / 2, y: size.height / 2)

                // 目标机位标记
                if let target {
                    TargetMarker(aligned: model.isAligned)
                        .position(target)
                        .animation(.interactiveSpring(response: 0.28, dampingFraction: 0.82), value: target)
                } else if model.targetNode != nil {
                    // 目标在视野外：边缘箭头
                    DirectionArrow(angle: arrowAngle(model: model))
                        .position(edgePosition(angle: arrowAngle(model: model), size: size))
                        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: arrowAngle(model: model))
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func reticleSize(aligned: Bool) -> CGFloat { aligned ? 96 : 116 }

    private func arrowAngle(model: CaptureModel) -> Angle {
        let yaw = model.targetOffsetYaw
        let pitch = model.targetOffsetPitch
        return Angle(radians: Double(atan2(-pitch, yaw)))
    }

    private func edgePosition(angle: Angle, size: CGSize) -> CGPoint {
        let inset: CGFloat = 46
        let rx = size.width / 2 - inset
        let ry = size.height / 2 - inset
        let c = CGFloat(cos(angle.radians))
        let s = CGFloat(sin(angle.radians))
        return CGPoint(x: size.width / 2 + c * rx,
                       y: size.height / 2 + s * ry)
    }

    /// 把目标方向投影到取景器坐标（不可见时返回 nil）
    private func projected(model: CaptureModel, size: CGSize) -> CGPoint? {
        guard let node = model.targetNode else { return nil }
        let cameraSpace = simd_mul(model.currentRotation.transpose, node.direction)
        guard cameraSpace.z < -0.02 else { return nil }
        let tx = cameraSpace.x / -cameraSpace.z
        let ty = cameraSpace.y / -cameraSpace.z
        let nx = tx / model.tanHalfH
        let ny = ty / model.tanHalfV
        guard abs(nx) < 1.4, abs(ny) < 1.4 else { return nil }
        let x = size.width / 2 + CGFloat(nx) * size.width / 2
        let y = size.height / 2 - CGFloat(ny) * size.height / 2
        return CGPoint(x: min(max(x, 20), size.width - 20),
                       y: min(max(y, 20), size.height - 20))
    }
}

private struct ReticleShape: Shape {
    var aligned: Bool

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let c = CGPoint(x: rect.midX, y: rect.midY)
        let r = min(rect.width, rect.height) / 2
        let gap: Double = aligned ? 0.62 : 0.42
        let outer = r
        let inner = r * (aligned ? 0.80 : 0.66)

        for i in 0..<4 {
            let start = Double(i) * Double.pi / 2 + Double.pi / 4 * gap
            let end = start + Double.pi / 2 * (1 - gap)
            path.addArc(center: c, radius: outer, startAngle: .radians(start),
                        endAngle: .radians(end), clockwise: false)
            let s2 = start
            let e2 = end
            path.move(to: CGPoint(x: c.x + cos(s2) * inner, y: c.y + sin(s2) * inner))
            path.addArc(center: c, radius: inner, startAngle: .radians(s2),
                        endAngle: .radians(e2), clockwise: false)
        }
        path.move(to: CGPoint(x: c.x - 6, y: c.y))
        path.addLine(to: CGPoint(x: c.x + 6, y: c.y))
        path.move(to: CGPoint(x: c.x, y: c.y - 6))
        path.addLine(to: CGPoint(x: c.x, y: c.y + 6))
        return path
    }
}

private struct TargetMarker: View {
    var aligned: Bool

    var body: some View {
        ZStack {
            Circle()
                .fill(Palette.mint.opacity(aligned ? 0.30 : 0.14))
                .frame(width: aligned ? 54 : 40, height: aligned ? 54 : 40)
            Circle()
                .strokeBorder(Palette.mint.opacity(aligned ? 0.95 : 0.7), lineWidth: 1.8)
                .frame(width: aligned ? 44 : 32, height: aligned ? 44 : 32)
            Image(systemName: "camera.aperture")
                .font(.system(size: aligned ? 15 : 12, weight: .semibold))
                .foregroundStyle(Palette.mint)
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: aligned)
    }
}

private struct DirectionArrow: View {
    var angle: Angle

    var body: some View {
        ZStack {
            Circle()
                .fill(.black.opacity(0.34))
                .overlay(Circle().strokeBorder(.white.opacity(0.18), lineWidth: 0.8))
                .frame(width: 44, height: 44)
            Image(systemName: "arrow.up")
                .font(.system(size: 17, weight: .bold))
                .foregroundStyle(.white)
                .rotationEffect(angle)
        }
    }
}
