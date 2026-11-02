import SwiftUI
import Photos

struct ResultView: View {
    @Environment(PanoramaStore.self) private var store
    let record: PanoramaRecord
    var onDone: () -> Void

    @State private var showSavedToast = false
    @State private var saving = false

    private var image: UIImage? { store.image(for: record) }

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(red: 0.05, green: 0.06, blue: 0.09),
                                    Color(red: 0.02, green: 0.02, blue: 0.03)],
                           startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()

            VStack(spacing: 16) {
                header
                    .padding(.horizontal, 20)
                    .padding(.top, 8)

                if let image {
                    PanoViewer(image: image)
                        .frame(maxHeight: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: 26, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 26, style: .continuous)
                                .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                        }
                        .padding(.horizontal, 12)
                        .overlay(alignment: .bottom) {
                            Text("拖动查看 · 双指缩放")
                                .font(.system(size: 10, weight: .medium, design: .rounded))
                                .foregroundStyle(.white.opacity(0.55))
                                .padding(.bottom, 12)
                        }
                } else {
                    ContentUnavailableView("图片丢失", systemImage: "photo.badge.exclamationmark")
                }

                stats
                    .padding(.horizontal, 20)

                actions
                    .padding(.horizontal, 20)
                    .padding(.bottom, 10)
            }

            if showSavedToast {
                VStack {
                    Spacer()
                    HStack(spacing: 7) {
                        Image(systemName: "checkmark.circle.fill")
                        Text("已保存到相册")
                    }
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .glass(Capsule(), tint: Palette.mint.opacity(0.4))
                    .padding(.bottom, 110)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                .allowsHitTesting(false)
            }
        }
        .preferredColorScheme(.dark)
    }

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 3) {
                Text("全景已生成")
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                Text(subtitleText)
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.55))
            }
            Spacer()
            Button(action: onDone) {
                Text("完成")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 9)
                    .glass(Capsule())
            }
            .buttonStyle(.plain)
        }
    }

    /// 标题下的一行小字：尺寸 / 投影 / 对齐残差（残差越小说明接缝对得越准）
    private var subtitleText: String {
        var s = "\(record.width) × \(record.height) · 等距圆柱投影"
        if let r = record.alignmentResidual, r > 0 {
            s += String(format: " · 对齐残差 %.2f°", r)
        }
        return s
    }

    private var stats: some View {        HStack(spacing: 10) {
            statItem(icon: "square.grid.3x3.fill",
                     title: "拼接帧数",
                     value: "\(record.shotCount)")
            statItem(icon: "circle.grid.cross.fill",
                     title: "球面覆盖",
                     value: String(format: "%.1f%%", record.coverage * 100))
            statItem(icon: "aspectratio.fill",
                     title: "比例",
                     value: String(format: "%.0f:1", record.aspect))
        }
    }

    private func statItem(icon: String, title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Palette.sky)
            Text(value)
                .font(.system(size: 15, weight: .semibold, design: .rounded))
                .foregroundStyle(.white)
                .monospacedDigit()
            Text(title)
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .foregroundStyle(.white.opacity(0.5))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 11)
        .glassCard(corner: 20)
    }

    private var actions: some View {
        HStack(spacing: 12) {
            Button {
                Task { await saveToPhotos() }
            } label: {
                HStack(spacing: 6) {
                    if saving {
                        ProgressView().tint(.white).scaleEffect(0.7)
                    } else {
                        Image(systemName: "square.and.arrow.down")
                            .font(.system(size: 13, weight: .bold))
                    }
                    Text("存到相册")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                }
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .glass(RoundedRectangle(cornerRadius: 18, style: .continuous), tint: Palette.sky.opacity(0.35))
            }
            .buttonStyle(.plain)
            .disabled(saving)

            if let url = imageURL {
                ShareLink(item: url) {
                    HStack(spacing: 6) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 13, weight: .bold))
                        Text("分享")
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                    }
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .glass(RoundedRectangle(cornerRadius: 18, style: .continuous))
                }
            }
        }
    }

    private var imageURL: URL? {
        let url = store.imageURL(for: record)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    private func saveToPhotos() async {
        guard let image else { return }
        saving = true
        defer { saving = false }
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else { return }
        do {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.creationRequestForAsset(from: image)
                } completionHandler: { success, error in
                    if success {
                        cont.resume()
                    } else {
                        cont.resume(throwing: error ?? StitchError.encode)
                    }
                }
            }
            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                showSavedToast = true
            }
            try? await Task.sleep(nanoseconds: 1_600_000_000)
            withAnimation(.easeOut(duration: 0.25)) {
                showSavedToast = false
            }
        } catch {
            // 忽略
        }
    }
}

/// 可拖动 / 缩放的全景查看器，左右无缝循环
struct PanoViewer: View {
    let image: UIImage

    @State private var offset: CGSize = .zero
    @State private var startOffset: CGSize = .zero
    @State private var zoom: CGFloat = 1
    @State private var startZoom: CGFloat = 1

    var body: some View {
        GeometryReader { geo in
            let viewW = geo.size.width
            let viewH = geo.size.height
            let h = viewH * zoom
            let aspect = image.size.width / max(1, image.size.height)
            let w = max(viewW, h * aspect)

            HStack(spacing: 0) {
                tile(width: w, height: h)
                tile(width: w, height: h)
            }
            .frame(width: w * 2, height: h, alignment: .leading)
            .offset(x: wrappedX(offset.width, width: w),
                    y: clampedY(offset.height, contentH: h, viewH: viewH))
            .frame(width: viewW, height: viewH, alignment: .topLeading)
            .clipped()
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 1)
                    .onChanged { value in
                        offset = CGSize(width: startOffset.width + value.translation.width,
                                        height: startOffset.height + value.translation.height)
                    }
                    .onEnded { _ in
                        offset.width = wrappedX(offset.width, width: w)
                        offset.height = clampedY(offset.height, contentH: h, viewH: viewH)
                        startOffset = offset
                    }
            )
            .simultaneousGesture(
                MagnifyGesture()
                    .onChanged { value in
                        zoom = min(4, max(1, startZoom * value.magnification))
                    }
                    .onEnded { _ in
                        startZoom = zoom
                        offset.width = wrappedX(offset.width, width: w)
                        startOffset = offset
                    }
            )
        }
    }

    private func tile(width: CGFloat, height: CGFloat) -> some View {
        Image(uiImage: image)
            .resizable()
            .interpolation(.high)
            .frame(width: width, height: height)
            .clipped()
    }

    private func wrappedX(_ x: CGFloat, width: CGFloat) -> CGFloat {
        guard width > 1 else { return 0 }
        var v = x.truncatingRemainder(dividingBy: width)
        if v > 0 { v -= width }
        return v
    }

    private func clampedY(_ y: CGFloat, contentH: CGFloat, viewH: CGFloat) -> CGFloat {
        let overflow = contentH - viewH
        if overflow <= 0 { return 0 }
        return min(0, max(-overflow, y))
    }
}
