import Foundation
import UIKit
import simd

/// 把上一次拍摄的原始帧 + 姿态留一份到 App 文档目录。
///
/// 目的：拼接效果如果不理想，可以直接从「文件」App 里把这一份取出来（App 已开启
/// `UIFileSharingEnabled`），在电脑上跑 `Tools/` 里的脚本离线复现，
/// 而不是靠截图去猜是哪一层出了问题。
///
/// 只保留最后一次（覆盖写），磁盘占用有上限。
enum DiagnosticsExport {

    static let folderName = "Pano720Debug"

    private struct FrameMeta: Codable {
        let file: String
        let rotation: [Float]      // 3x3 行主序，相机 -> 世界
        let tanHalfH: Float
        let tanHalfV: Float
        let crop: Float
        let luma: Float
        let nodeID: Int?
    }

    private struct SessionMeta: Codable {
        let createdAt: String
        let shotCount: Int
        let aligner: String
        let calibration: String       // 真机实测出的图像朝向 / 焦距修正
        let rotationNote: String      // 说明 frames.rotation 是校准前还是校准后
        let residualDeg: Double
        let pairCount: Int
        let coverage: Double
        let outWidth: Int
        let outHeight: Int
        let seconds: Double
        let frames: [FrameMeta]
    }

    @discardableResult
    static func export(shots: [CaptureModel.Shot],
                       sessionDir: URL,
                       panorama: StitchedPanorama) -> URL? {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        let root = docs.appendingPathComponent(folderName, isDirectory: true)
        let dest = root.appendingPathComponent("last", isDirectory: true)

        try? fm.removeItem(at: dest)
        guard (try? fm.createDirectory(at: dest, withIntermediateDirectories: true)) != nil else {
            return nil
        }

        var metas: [FrameMeta] = []
        metas.reserveCapacity(shots.count)

        for (index, shot) in shots.enumerated() {
            let src = sessionDir.appendingPathComponent(shot.fileName)
            let name = String(format: "frame-%03d.jpg", index)
            // 帧写盘失败就跳过，不要因为某一帧缺失让整份诊断失败
            guard (try? fm.copyItem(at: src, to: dest.appendingPathComponent(name))) != nil else {
                continue
            }
            let m = shot.rotation
            var rows: [Float] = []
            rows.reserveCapacity(9)
            for r in 0..<3 {
                for c in 0..<3 {
                    rows.append(m[c][r])       // 行主序
                }
            }
            metas.append(FrameMeta(file: name,
                                   rotation: rows,
                                   tanHalfH: shot.intrinsics.tanHalfH,
                                   tanHalfV: shot.intrinsics.tanHalfV,
                                   crop: shot.intrinsics.crop,
                                   luma: shot.luma,
                                   nodeID: shot.nodeID))
        }

        let formatter = ISO8601DateFormatter()
        let meta = SessionMeta(createdAt: formatter.string(from: Date()),
                               shotCount: shots.count,
                               aligner: panorama.aligner,
                               calibration: panorama.calibration,
                               rotationNote: "frames[].rotation 是陀螺仪原始姿态；拼接时右乘了 calibration 里的朝向矩阵 Qᵀ",
                               residualDeg: panorama.residualDeg,
                               pairCount: panorama.pairCount,
                               coverage: panorama.coverage,
                               outWidth: panorama.width,
                               outHeight: panorama.height,
                               seconds: panorama.seconds,
                               frames: metas)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(meta) {
            try? data.write(to: dest.appendingPathComponent("session.json"), options: .atomic)
        }
        // 顺带把成品也放进来，方便和帧放在一起对照
        try? panorama.jpeg.write(to: dest.appendingPathComponent("panorama.jpg"), options: .atomic)

        return dest
    }
}
