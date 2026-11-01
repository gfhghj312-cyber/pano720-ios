import SwiftUI

struct HomeView: View {
    @Environment(PanoramaStore.self) private var store
    @State private var showCapture = false
    @State private var selected: PanoramaRecord?

    private let columns = [GridItem(.flexible(), spacing: 12),
                           GridItem(.flexible(), spacing: 12)]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    heroCard
                    library
                }
                .padding(.horizontal, 18)
                .padding(.top, 6)
                .padding(.bottom, 40)
            }
            .background(background.ignoresSafeArea())
            .navigationTitle("全景 720")
            .toolbarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Text("\(store.items.count) 张")
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(.secondary)
                }
            }
            .fullScreenCover(isPresented: $showCapture) {
                CaptureView()
            }
            .fullScreenCover(item: $selected) { record in
                ResultView(record: record) {
                    selected = nil
                }
            }
        }
    }

    private var background: some View {
        LinearGradient(colors: [Color(uiColor: .systemBackground),
                                Palette.violet.opacity(0.12),
                                Palette.sky.opacity(0.06)],
                       startPoint: .top,
                       endPoint: .bottom)
    }

    // MARK: - 主卡片

    private var heroCard: some View {
        Button {
            showCapture = true
        } label: {
            ZStack(alignment: .bottomLeading) {
                RoundedRectangle(cornerRadius: 30, style: .continuous)
                    .fill(Palette.aurora)
                    .overlay(alignment: .topTrailing) {
                        SphereDecoration()
                            .frame(width: 210, height: 210)
                            .offset(x: 34, y: -46)
                    }
                    .overlay(alignment: .bottomLeading) {
                        VStack(alignment: .leading, spacing: 7) {
                            Text("拍摄 720° 全景")
                                .font(.system(size: 22, weight: .bold, design: .rounded))
                            Text("转动手机跟随指引，自动对齐 · 自动拼接")
                                .font(.system(size: 12, weight: .medium, design: .rounded))
                                .opacity(0.82)
                            HStack(spacing: 6) {
                                Image(systemName: "camera.fill")
                                    .font(.system(size: 11, weight: .bold))
                                Text("开始拍摄")
                                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 9)
                            .background(Capsule().fill(.white.opacity(0.22)))
                            .overlay(Capsule().strokeBorder(.white.opacity(0.35), lineWidth: 0.8))
                            .padding(.top, 4)
                        }
                        .foregroundStyle(.white)
                        .padding(20)
                    }
                    .frame(height: 190)
                    .shadow(color: Palette.violet.opacity(0.35), radius: 22, y: 12)
            }
        }
        .buttonStyle(.plain)
    }

    // MARK: - 相册

    @ViewBuilder
    private var library: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("我的全景")
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                Spacer()
                if !store.items.isEmpty {
                    Text("长按可删除")
                        .font(.system(size: 10, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 4)

            if store.items.isEmpty {
                ContentUnavailableView {
                    Label("还没有全景", systemImage: "pano")
                } description: {
                    Text("点上面的按钮，拍下第一张 720° 全景")
                }
                .padding(.vertical, 26)
            } else {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(store.items) { record in
                        Button {
                            selected = record
                        } label: {
                            card(record)
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button(role: .destructive) {
                                store.delete(record)
                            } label: {
                                Label("删除", systemImage: "trash")
                            }
                        }
                    }
                }
            }
        }
    }

    private func card(_ record: PanoramaRecord) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(Color(uiColor: .secondarySystemBackground))
                if let image = store.thumbnail(for: record) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                } else {
                    Image(systemName: "pano")
                        .font(.system(size: 22))
                        .foregroundStyle(.secondary)
                }
            }
            .frame(height: 118)
            .overlay(alignment: .topTrailing) {
                Text(String(format: "%.0f%%", record.coverage * 100))
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(.black.opacity(0.45)))
                    .padding(7)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(record.title)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                Text("\(record.shotCount) 帧 · \(record.width)×\(record.height)")
                    .font(.system(size: 9, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 2)
        }
    }
}

/// 主卡片右上角的球面装饰
private struct SphereDecoration: View {
    var body: some View {
        Canvas { ctx, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let r = min(size.width, size.height) / 2 - 6
            var circle = Path()
            circle.addEllipse(in: CGRect(x: center.x - r, y: center.y - r, width: r * 2, height: r * 2))
            ctx.stroke(circle, with: .color(.white.opacity(0.55)), lineWidth: 1.2)

            for i in 1..<4 {
                let k = CGFloat(i) / 4
                var lat = Path()
                lat.addEllipse(in: CGRect(x: center.x - r, y: center.y - r * k,
                                          width: r * 2, height: r * 2 * k))
                ctx.stroke(lat, with: .color(.white.opacity(0.28)), lineWidth: 0.8)

                var lon = Path()
                lon.addEllipse(in: CGRect(x: center.x - r * k, y: center.y - r,
                                          width: r * 2 * k, height: r * 2))
                ctx.stroke(lon, with: .color(.white.opacity(0.28)), lineWidth: 0.8)
            }

            for i in 0..<8 {
                let a = CGFloat(i) * .pi / 4
                var line = Path()
                line.move(to: center)
                line.addLine(to: CGPoint(x: center.x + cos(a) * r, y: center.y + sin(a) * r))
                ctx.stroke(line, with: .color(.white.opacity(0.14)), lineWidth: 0.6)
            }
        }
        .blendMode(.softLight)
    }
}
