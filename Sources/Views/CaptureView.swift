import SwiftUI

struct CaptureView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(PanoramaStore.self) private var store

    @State private var model = CaptureModel()
    @State private var flashOpacity: Double = 0
    @State private var savedRecord: PanoramaRecord?
    @State private var pendingStitch = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                topBar
                    .padding(.horizontal, 16)
                    .padding(.top, 4)

                sphereCard
                    .padding(.horizontal, 30)
                    .padding(.top, 12)

                Spacer(minLength: 10)

                viewfinder

                Spacer(minLength: 10)

                hintPill
                    .padding(.horizontal, 24)

                bottomBar
                    .padding(.horizontal, 26)
                    .padding(.top, 14)
                    .padding(.bottom, 8)
            }

            Color.white
                .opacity(flashOpacity)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            if case .stitching = model.phase {
                StitchOverlay(progress: model.stitchProgress)
                    .transition(.opacity)
            }

            if case .failed(let message) = model.phase {
                FailureOverlay(message: message) {
                    dismiss()
                }
            }
        }
        .preferredColorScheme(.dark)
        .confirmationDialog("还有 \(max(0, model.requiredCount - model.shots.count)) 个方向没拍",
                            isPresented: $pendingStitch,
                            titleVisibility: .visible) {
            Button("继续拍摄", role: .cancel) {}
            Button("仍然拼接", role: .destructive) {
                Task { await finish() }
            }
        } message: {
            Text("已拍 \(model.shots.count) 张，这个机位配置需要约 \(model.requiredCount) 张才能覆盖整圈。没拍到的方向会留空，重叠不够的地方会错位。")
        }
        .task {
            model.start()
        }
        .onDisappear {
            model.stop()
        }
        .onChange(of: model.flashTick) { _, _ in
            flashOpacity = 0.62
            withAnimation(.easeOut(duration: 0.24)) {
                flashOpacity = 0
            }
        }
        .fullScreenCover(item: $savedRecord, onDismiss: {
            dismiss()
        }) { record in
            ResultView(record: record) {
                savedRecord = nil
            }
        }
    }

    // MARK: - 顶部

    private var topBar: some View {
        HStack(spacing: 10) {
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 38, height: 38)
                    .glass(Circle())
            }
            .buttonStyle(.plain)

            Spacer(minLength: 0)

            HStack(spacing: 8) {
                ProgressRing(progress: model.progress, lineWidth: 3.4)
                    .frame(width: 20, height: 20)
                Text("\(model.capturedCount)/\(model.nodes.count)")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(.white)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .glass(Capsule())

            Spacer(minLength: 0)

            Button {
                if shouldWarnIncomplete {
                    pendingStitch = true
                } else {
                    Task { await finish() }
                }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "wand.and.stars")
                        .font(.system(size: 12, weight: .bold))
                    Text("拼接")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                }
                .foregroundStyle(model.shots.count >= 2 ? Color.white : Color.white.opacity(0.35))
                .padding(.horizontal, 13)
                .padding(.vertical, 9)
                .glass(Capsule(), tint: model.shots.count >= 2 ? Palette.violet.opacity(0.55) : nil)
            }
            .buttonStyle(.plain)
            .disabled(model.shots.count < 2)
        }
    }

    private var sphereCard: some View {
        SphereMap(nodes: model.nodes, currentForward: model.currentForward)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .glassCard(corner: 20)
    }

    // MARK: - 取景器

    private var viewfinder: some View {
        CameraPreview(session: model.camera.session)
            .aspectRatio(model.imageAspect, contentMode: .fit)
            .overlay {
                AlignmentReticle(model: model)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 30, style: .continuous)
                    .strokeBorder(.white.opacity(0.12), lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: 30, style: .continuous))
            .padding(.horizontal, 10)
            .overlay(alignment: .topLeading) {
                lensPill
                    .padding(.leading, 20)
                    .padding(.top, 14)
            }
            .overlay(alignment: .bottomTrailing) {
                if let thumb = model.lastThumbnail {
                    Image(uiImage: thumb)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 42, height: 42)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .strokeBorder(.white.opacity(0.5), lineWidth: 1)
                        }
                        .padding(.trailing, 20)
                        .padding(.bottom, 12)
                        .transition(.scale.combined(with: .opacity))
                }
            }
    }

    private var lensPill: some View {
        HStack(spacing: 2) {
            ForEach(CameraLens.allCases) { item in
                let enabled = item != .ultraWide || model.supportsUltraWide
                Button {
                    model.switchLens(item)
                } label: {
                    Text(item.shortLabel)
                        .font(.system(size: 12, weight: .bold, design: .rounded))
                        .foregroundStyle(model.lens == item ? Color.black : Color.white.opacity(0.82))
                        .frame(width: 44, height: 34)
                        .background {
                            if model.lens == item {
                                Capsule().fill(.white)
                            }
                        }
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .disabled(!enabled)
                .opacity(enabled ? 1 : 0.28)
            }
        }
        .padding(3)
        .glass(Capsule())
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: model.lens)
        .overlay(alignment: .bottomLeading) {
            if model.isSwitchingLens {
                ProgressView()
                    .controlSize(.mini)
                    .tint(.white)
                    .offset(x: 6, y: -6)
            }
        }
    }

    private var hintPill: some View {
        Text(model.notice ?? model.hintText)
            .font(.system(size: 13, weight: .medium, design: .rounded))
            .foregroundStyle(model.notice != nil ? Palette.amber : hintColor)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .glass(Capsule())
            .animation(.easeInOut(duration: 0.2), value: model.hintText)
            .animation(.easeInOut(duration: 0.2), value: model.notice)
    }

    private var hintColor: Color {
        if model.isAligned && model.isSteady { return Palette.mint }
        if model.isSteady { return .white.opacity(0.85) }
        return Palette.amber
    }

    // MARK: - 底部

    private var bottomBar: some View {
        ZStack {
            HStack {
                Button {
                    model.undoLast()
                } label: {
                    Image(systemName: "arrow.uturn.backward")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.white.opacity(model.shots.isEmpty ? 0.3 : 0.95))
                        .frame(width: 48, height: 48)
                        .glass(Circle())
                }
                .buttonStyle(.plain)
                .disabled(model.shots.isEmpty)

                Spacer()

                Button {
                    model.toggleAutoCapture()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: model.isAutoCaptureEnabled ? "bolt.fill" : "bolt.slash")
                            .font(.system(size: 13, weight: .bold))
                        Text(model.isAutoCaptureEnabled ? "自动" : "手动")
                            .font(.system(size: 12, weight: .semibold, design: .rounded))
                    }
                    .foregroundStyle(model.isAutoCaptureEnabled ? Palette.mint : .white.opacity(0.6))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 13)
                    .glass(Capsule(),
                           tint: model.isAutoCaptureEnabled ? Palette.mint.opacity(0.35) : nil)
                }
                .buttonStyle(.plain)
            }

            Button {
                model.manualCapture()
            } label: {
                ZStack {
                    Circle()
                        .strokeBorder(.white.opacity(0.85), lineWidth: 2.5)
                        .frame(width: 76, height: 76)
                    Circle()
                        .fill(model.isAligned ? Palette.mint : Color.white)
                        .frame(width: 62, height: 62)
                        .shadow(color: model.isAligned ? Palette.mint.opacity(0.6) : .clear, radius: 12)
                }
            }
            .buttonStyle(ShutterButtonStyle())
            .disabled(model.phase != .ready)
        }
    }

    // MARK: - 动作

    private func finish() async {
        do {
            let panorama = try await model.stitch()
            if let record = store.save(panorama, shotCount: model.shots.count) {
                // 先导出一份原始帧 + 姿态，再清临时文件（诊断用，只留最后一次）
                model.exportDiagnostics(for: panorama)
                model.cleanupWorkingFiles()
                savedRecord = record
            }
        } catch {
            // model.phase 已带上错误信息
        }
    }

    /// 拍摄不足时先拦一下：没覆盖到的方向拼出来必然是空白 + 错位，
    /// 直接拼会让人以为是拼接算法坏了。
    private var shouldWarnIncomplete: Bool {
        model.shots.count >= 2 && model.shots.count * 100 < model.requiredCount * 85
    }
}

private struct ShutterButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.9 : 1)
            .animation(.spring(response: 0.25, dampingFraction: 0.65), value: configuration.isPressed)
    }
}

// MARK: - 全屏覆盖层

private struct StitchOverlay: View {
    var progress: Double

    var body: some View {
        ZStack {
            Color.black.opacity(0.55).ignoresSafeArea()
            VStack(spacing: 18) {
                ZStack {
                    ProgressRing(progress: max(0.02, progress), lineWidth: 6)
                        .frame(width: 74, height: 74)
                    Text("\(Int(progress * 100))%")
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                        .monospacedDigit()
                        .foregroundStyle(.white)
                }
                Text("正在拼接 720° 全景")
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                Text("对齐机位 · 羽化融合 · 亮度校正")
                    .font(.system(size: 11, weight: .regular, design: .rounded))
                    .foregroundStyle(.white.opacity(0.6))
            }
            .padding(.horizontal, 30)
            .padding(.vertical, 26)
            .glassCard(corner: 30)
        }
    }
}

private struct FailureOverlay: View {
    var message: String
    var onClose: () -> Void

    var body: some View {
        ZStack {
            Color.black.opacity(0.6).ignoresSafeArea()
            VStack(spacing: 14) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 26))
                    .foregroundStyle(Palette.rose)
                Text(message)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.9))
                    .multilineTextAlignment(.center)
                Button(action: onClose) {
                    Text("返回")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 22)
                        .padding(.vertical, 10)
                        .glass(Capsule())
                }
                .buttonStyle(.plain)
            }
            .padding(24)
            .frame(maxWidth: 300)
            .glassCard(corner: 28)
        }
    }
}
