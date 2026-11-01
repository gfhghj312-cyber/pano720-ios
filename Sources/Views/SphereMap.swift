import SwiftUI
import simd

/// 球面展开小地图：直观看到哪些机位已经拍过、当前朝向在哪
struct SphereMap: View {
    var nodes: [CaptureNode]
    var currentForward: Vec3
    var accent: Color = Palette.mint

    var body: some View {
        Canvas { ctx, size in
            guard !nodes.isEmpty, size.width > 1 else { return }
            let base = nodes[0].yaw

            func point(_ pitch: Float, _ yaw: Float) -> CGPoint {
                let rel = wrapAngle(yaw - base)
                let x = (CGFloat(rel) + .pi) / (2 * .pi) * size.width
                let y = (0.5 - CGFloat(pitch) / .pi) * size.height
                return CGPoint(x: x, y: y)
            }

            var horizon = Path()
            horizon.move(to: CGPoint(x: 0, y: size.height / 2))
            horizon.addLine(to: CGPoint(x: size.width, y: size.height / 2))
            ctx.stroke(horizon, with: .color(.white.opacity(0.14)),
                       style: StrokeStyle(lineWidth: 0.7, dash: [3, 5]))

            for node in nodes {
                let p = point(node.pitch, node.yaw)
                let r: CGFloat = node.kind == .ring ? 2.7 : 2.2
                let rect = CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)
                if node.captured {
                    ctx.fill(Path(ellipseIn: rect), with: .color(accent))
                } else {
                    ctx.stroke(Path(ellipseIn: rect.insetBy(dx: -0.5, dy: -0.5)),
                               with: .color(.white.opacity(node.isOptional ? 0.22 : 0.5)),
                               lineWidth: 0.9)
                }
            }

            let c = point(pitchOf(currentForward), yawOf(currentForward))
            var cross = Path()
            cross.move(to: CGPoint(x: c.x, y: max(1, c.y - 6)))
            cross.addLine(to: CGPoint(x: c.x, y: min(size.height - 1, c.y + 6)))
            cross.move(to: CGPoint(x: max(1, c.x - 4), y: c.y))
            cross.addLine(to: CGPoint(x: min(size.width - 1, c.x + 4), y: c.y))
            ctx.stroke(cross, with: .color(.white.opacity(0.95)), lineWidth: 1.3)
        }
        .frame(height: 62)
    }
}
