import SwiftUI

enum Palette {
    static let mint = Color(red: 0.30, green: 0.90, blue: 0.72)      // 已对齐
    static let sky = Color(red: 0.36, green: 0.78, blue: 1.00)       // 主色
    static let violet = Color(red: 0.58, green: 0.52, blue: 1.00)
    static let amber = Color(red: 1.00, green: 0.74, blue: 0.33)     // 提示
    static let rose = Color(red: 1.00, green: 0.45, blue: 0.48)

    static let aurora = LinearGradient(colors: [Palette.sky, Palette.violet],
                                       startPoint: .topLeading,
                                       endPoint: .bottomTrailing)

    static let progress = LinearGradient(colors: [Palette.mint, Palette.sky],
                                         startPoint: .leading,
                                         endPoint: .trailing)
}

// MARK: - Liquid Glass
// iOS 26+ 用系统原生 glassEffect；更低版本回退到 ultraThinMaterial + 高光描边，
// 视觉上保持同一套「玻璃层 + 同心圆角」语言。

struct GlassSurface<S: Shape>: ViewModifier {
    var shape: S
    var tint: Color?
    var interactive: Bool

    @ViewBuilder
    func body(content: Content) -> some View {
        #if compiler(>=6.2)
        if #available(iOS 26.0, *) {
            let glass: Glass = {
                var g = Glass.regular
                if let tint { g = g.tint(tint) }
                if interactive { g = g.interactive() }
                return g
            }()
            content.glassEffect(glass, in: shape)
        } else {
            legacy(content)
        }
        #else
        legacy(content)
        #endif
    }

    @ViewBuilder
    private func legacy(_ content: Content) -> some View {
        content
            .background {
                shape
                    .fill(.ultraThinMaterial)
                    .overlay {
                        shape
                            .strokeBorder(
                                LinearGradient(colors: [.white.opacity(0.42), .white.opacity(0.06)],
                                               startPoint: .topLeading,
                                               endPoint: .bottomTrailing),
                                lineWidth: 0.9
                            )
                    }
                    .shadow(color: .black.opacity(0.28), radius: 16, y: 6)
            }
    }
}

extension View {
    func glass<S: Shape>(_ shape: S,
                         tint: Color? = nil,
                         interactive: Bool = false) -> some View {
        modifier(GlassSurface(shape: shape, tint: tint, interactive: interactive))
    }

    func glassCard(corner: CGFloat = 28, tint: Color? = nil) -> some View {
        glass(RoundedRectangle(cornerRadius: corner, style: .continuous), tint: tint)
    }
}

// MARK: - 通用组件

struct ProgressRing: View {
    var progress: Double
    var lineWidth: CGFloat = 5
    var gradient: LinearGradient = Palette.progress

    var body: some View {
        ZStack {
            Circle()
                .strokeBorder(Color.white.opacity(0.16), lineWidth: lineWidth)
            Circle()
                .trim(from: 0, to: max(0.001, min(1, progress)))
                .stroke(gradient, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.spring(response: 0.45, dampingFraction: 0.8), value: progress)
        }
    }
}

struct CapsuleTag: View {
    var text: String
    var icon: String?
    var tint: Color = .white

    var body: some View {
        HStack(spacing: 5) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 10, weight: .bold))
            }
            Text(text)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .monospacedDigit()
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Capsule().fill(Color.black.opacity(0.28)))
        .overlay(Capsule().strokeBorder(Color.white.opacity(0.14), lineWidth: 0.8))
    }
}
