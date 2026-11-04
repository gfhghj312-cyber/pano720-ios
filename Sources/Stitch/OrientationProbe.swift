import Foundation
import simd

/// 图像朝向的 8 种可能（D4 群）。
///
/// 为什么需要它：`CGBitmapContext` 的行序、`videoRotationAngle` 是否真的旋转了像素、
/// `encodeJPEG` 里的 `CIImage.oriented` 到底转了哪个方向、`videoFieldOfView` 描述长边还是短边
/// —— 这些在真机上的实际行为无法在桌面端验证，连续几轮靠推断写死都写错了。
/// 现在一律不猜：把 8 种可能全部当候选，用重叠区一致度在设备上实测打分。
///
/// 实现上的关键技巧：Q 只置换/取反相机系的 (x, y)，保持 z（光轴）不变。
/// 把它右乘进姿态 `R' = R·Qᵀ` 后，着色器里的 `c = R'ᵀ·w` 就自动变成 `Q·(Rᵀw)`，
/// 于是 `u, v` 会落在实际图像的正确像素上。Metal / OpenCV / 光度配准三条路径
/// 拿到的都是同一个 `rotation`，无需各自处理方向，也不存在「配准看到正帧、拼接拿到倒帧」。
enum AxisCalibration: Int, CaseIterable {
    case identity = 0
    case flipY          // 图像上下颠倒
    case flipX          // 图像左右镜像
    case rot180
    case rot90cw        // 图像轴与相机轴差 90°
    case rot90ccw
    case transpose
    case antiTranspose

    var label: String {
        switch self {
        case .identity: return "原始"
        case .flipY: return "上下翻转"
        case .flipX: return "左右镜像"
        case .rot180: return "旋转 180°"
        case .rot90cw: return "旋转 90°"
        case .rot90ccw: return "旋转 270°"
        case .transpose: return "转置镜像"
        case .antiTranspose: return "反置镜像"
        }
    }

    /// Q：作用于相机系向量，只动 x/y。第 2 列恒为 (0,0,1)，所以 cameraForward 不受影响。
    /// simd_float3x3 的构造参数是「列」。
    var matrix: simd_float3x3 {
        switch self {
        case .identity:
            return simd_float3x3(SIMD3<Float>( 1,  0, 0), SIMD3<Float>( 0,  1, 0), SIMD3<Float>(0, 0, 1))
        case .flipY:
            return simd_float3x3(SIMD3<Float>( 1,  0, 0), SIMD3<Float>( 0, -1, 0), SIMD3<Float>(0, 0, 1))
        case .flipX:
            return simd_float3x3(SIMD3<Float>(-1,  0, 0), SIMD3<Float>( 0,  1, 0), SIMD3<Float>(0, 0, 1))
        case .rot180:
            return simd_float3x3(SIMD3<Float>(-1,  0, 0), SIMD3<Float>( 0, -1, 0), SIMD3<Float>(0, 0, 1))
        case .rot90cw:      // ax = cy, ay = -cx
            return simd_float3x3(SIMD3<Float>( 0, -1, 0), SIMD3<Float>( 1,  0, 0), SIMD3<Float>(0, 0, 1))
        case .rot90ccw:     // ax = -cy, ay = cx
            return simd_float3x3(SIMD3<Float>( 0,  1, 0), SIMD3<Float>(-1,  0, 0), SIMD3<Float>(0, 0, 1))
        case .transpose:    // ax = cy, ay = cx
            return simd_float3x3(SIMD3<Float>( 0,  1, 0), SIMD3<Float>( 1,  0, 0), SIMD3<Float>(0, 0, 1))
        case .antiTranspose: // ax = -cy, ay = -cx
            return simd_float3x3(SIMD3<Float>( 0, -1, 0), SIMD3<Float>(-1,  0, 0), SIMD3<Float>(0, 0, 1))
        }
    }
}

struct CalibrationInfo {
    let orientation: AxisCalibration
    let focalScale: Float
    let score: Float
    let baseline: Float
    let runnerUp: Float

    /// 结果页的一行小字，用来确认校准是否真的生效
    var display: String {
        guard baseline > 0 else { return "朝向校准：数据不足" }
        let gain = Double(max(0, (1 - score / baseline) * 100))
        let f = abs(focalScale - 1) > 0.005
            ? String(format: " · 焦距 %+.0f%%", Double((focalScale - 1) * 100)) : ""
        let margin = score > 0 ? Double(runnerUp / score) : 1
        return String(format: "朝向校准：%@%@ · 重影 %.0f → %.0f（%.0f%%↓）· 置信 %.1f×",
                      orientation.label, f,
                      Double(baseline), Double(score), gain, margin)
    }
}

enum OrientationProbe {

    struct Outcome {
        let info: CalibrationInfo
        let inputs: [StitchInput]
    }

    /// 真机上用重叠区一致度自动选出正确朝向与焦距比例。
    ///
    /// 判据与全局姿态误差无关：两帧在同一球面方向采到的像素本就来自同一物理点，
    /// 整张全景统一转个角度不改变这个分数。所以它能干净地只反映
    /// 「相对姿态 + 内参 + 图像朝向」。Tools/verify_orientation_probe.py 里
    /// 8 种朝向注入 8/8 选中正确项，分数 ~8（对） vs ~55–65（错）。
    static func calibrate(_ inputs: [StitchInput]) -> Outcome? {
        guard inputs.count >= 3 else { return nil }

        // 用小灰度图打分。它和 Metal / OpenCV 走的是同一套 drawTopDown 解码约定，
        // 所以「identity」候选就代表当前工程的行序基准；探测会补偿任何基准错误。
        var frames: [FrameAligner.Frame] = []
        frames.reserveCapacity(inputs.count)
        for input in inputs {
            guard let pyr = FrameAligner.pyramid(url: input.url, shortEdge: 288, levels: 1) else { return nil }
            frames.append(FrameAligner.Frame(rotation: input.rotation,
                                             tanH: input.tanHalfH,
                                             tanV: input.tanHalfV,
                                             crop: input.crop,
                                             pyramid: pyr))
        }

        let baseH = inputs.map(\.tanHalfH)
        let baseV = inputs.map(\.tanHalfV)

        func score(_ q: simd_float3x3, _ k: Float) -> Float {
            let qt = q.transpose
            let rots = inputs.map { simd_mul($0.rotation, qt) }
            let th = baseH.map { max(0.05, $0 * k) }
            let tv = baseV.map { max(0.05, $0 * k) }
            return FrameAligner.ghostScore(frames: frames, rotations: rots,
                                           tanH: th, tanV: tv)
        }

        let baseline = score(AxisCalibration.identity.matrix, 1)

        // 1) 8 个朝向
        var axisScores: [(AxisCalibration, Float)] = []
        for axis in AxisCalibration.allCases {
            axisScores.append((axis, score(axis.matrix, 1)))
        }
        axisScores.sort { a, b in
            let x = a.1 < 0 ? Float.greatestFiniteMagnitude : a.1
            let y = b.1 < 0 ? Float.greatestFiniteMagnitude : b.1
            return x < y
        }
        guard let best = axisScores.first else { return nil }
        let bestAxis = best.0
        let bestScore = best.1
        let runnerUp: Float = {
            guard axisScores.count > 1, axisScores[1].1 > 0 else { return best.1 }
            return axisScores[1].1
        }()
        if bestScore < 0 { return nil }

        // 2) 在选中的朝向上粗扫焦距比例。videoFieldOfView 报偏 10% 就会让
        //    接缝从中心往外散开，这是纯几何手段能测到的另一半标定误差。
        let scales: [Float] = [0.82, 0.88, 0.94, 1.0, 1.07, 1.15, 1.24]
        var bestScale: Float = 1
        var bestFocalScore = bestScore
        for k in scales where k != 1 {
            let s = score(bestAxis.matrix, k)
            if s >= 0 && s < bestFocalScore * 0.97 {
                bestFocalScore = s
                bestScale = k
            }
        }

        guard bestScore > 0 || bestFocalScore > 0 else { return nil }

        let q = bestAxis.matrix.transpose
        let corrected: [StitchInput] = inputs.map { input in
            StitchInput(url: input.url,
                        rotation: simd_mul(input.rotation, q),
                        tanHalfH: max(0.05, input.tanHalfH * bestScale),
                        tanHalfV: max(0.05, input.tanHalfV * bestScale),
                        crop: input.crop,
                        luma: input.luma)
        }

        let info = CalibrationInfo(orientation: bestAxis,
                                   focalScale: bestScale,
                                   score: min(bestScore, bestFocalScore),
                                   baseline: baseline,
                                   runnerUp: runnerUp)
        return Outcome(info: info, inputs: corrected)
    }
}
