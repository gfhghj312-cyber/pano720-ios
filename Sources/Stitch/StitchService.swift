import Foundation
import UIKit
import Metal
import CoreGraphics
import ImageIO
import simd

struct StitchInput {
    let url: URL
    let rotation: simd_float3x3   // 相机 -> 世界
    let tanHalfH: Float           // 全画幅 竖屏水平半视场角正切
    let tanHalfV: Float           // 全画幅 竖屏垂直半视场角正切
    let crop: Float               // 参与拼接的中央比例
    let luma: Float
}

struct StitchedPanorama {
    let equirect: CGImage
    let jpeg: Data
    let thumbnail: Data
    let coverage: Double
    let seconds: TimeInterval
    let width: Int
    let height: Int
    /// 精配准后的残余姿态残差（度），越小说明重叠区对得越准
    let residualDeg: Double
    let pairCount: Int
}

enum StitchError: LocalizedError {
    case noDevice
    case noInput
    case shader(String)
    case texture(String)
    case encode

    var errorDescription: String? {
        switch self {
        case .noDevice: return "此设备不支持 Metal"
        case .noInput: return "没有可拼接的画面"
        case .shader(let m): return "着色器编译失败：\(m)"
        case .texture(let m): return "纹理创建失败：\(m)"
        case .encode: return "图像编码失败"
        }
    }
}

/// 基于姿态（陀螺仪）+ 光度精配准的球面投影拼接。
///
/// 1. `FrameAligner` 用重叠区光度误差修正每帧姿态（陀螺仪漂移会让接缝出现双重边缘）。
/// 2. 对等距圆柱目标图的每个像素求世界射线 w，用每帧旋转算出相机系 c = Rᵀ·w，
///    做针孔反投影采样，权重 = 边缘羽化 × cos³(离轴角) × 逐帧增益。
/// 3. 第二轮对没覆盖到的像素做「放宽边界 + 钳制采样」的补洞。
enum StitchService {

    // MARK: - Metal 着色器（运行时编译，免去 .metal 构建配置）

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Header {
        int outW;
        int outH;
        int count;
        int pad;
    };

    struct Img {
        float4 r0;     // Rᵀ 的第 0 行（世界 -> 相机）
        float4 r1;     // Rᵀ 的第 1 行
        float4 r2;     // Rᵀ 的第 2 行
        float4 misc;   // x: tanHalfH, y: tanHalfV, z: gain, w: slice
        float4 ext;    // x: 覆盖用 tanH, y: 覆盖用 tanV, z: 0.5*crop, w: feather 宽度
    };

    // 相机看 -Z：可见点在相机系里 z < 0
    static inline float3 toCamera(Img im, float3 w) {
        return float3(dot(im.r0.xyz, w), dot(im.r1.xyz, w), dot(im.r2.xyz, w));
    }

    kernel void stitchPano(texture2d_array<float, access::sample> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           constant Header &hp [[buffer(0)]],
                           constant Img *imgs [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= uint(hp.outW) || gid.y >= uint(hp.outH)) { return; }

        constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);

        float u = (float(gid.x) + 0.5) / float(hp.outW);
        float v = (float(gid.y) + 0.5) / float(hp.outH);
        float lon = (u - 0.5) * 6.28318530718;
        float lat = (0.5 - v) * 3.14159265359;
        float cl = cos(lat);
        float3 w = float3(cl * cos(lon), cl * sin(lon), sin(lat));

        float3 acc = float3(0.0);
        float wsum = 0.0;

        // 第一轮：严格落在画幅内，按到裁切边长度的距离羽化
        for (int i = 0; i < hp.count; ++i) {
            Img im = imgs[i];
            float3 c = toCamera(im, w);
            if (c.z >= -0.0001) { continue; }
            float inv = 1.0 / (-c.z);
            float tx = c.x * inv;
            float ty = c.y * inv;
            if (fabs(tx) > im.ext.x || fabs(ty) > im.ext.y) { continue; }

            float su = 0.5 + 0.5 * tx / im.misc.x;
            float sv = 0.5 - 0.5 * ty / im.misc.y;
            if (su < 0.0 || su > 1.0 || sv < 0.0 || sv > 1.0) { continue; }

            float eu = im.ext.z - fabs(su - 0.5);
            float ev = im.ext.z - fabs(sv - 0.5);
            if (eu <= 0.0 || ev <= 0.0) { continue; }

            float feather = smoothstep(0.0, im.ext.w, eu) * smoothstep(0.0, im.ext.w, ev);
            if (feather <= 0.00001) { continue; }

            // 平方一下：过渡带更集中，接缝更"窄"，不会把两帧的轻微错位摊成大片半透明重影
            float cosA2 = 1.0 / (1.0 + tx * tx + ty * ty);
            float wt = feather * feather * cosA2 * sqrt(cosA2) * im.misc.z;

            float4 col = src.sample(smp, float2(su, sv), uint(im.misc.w + 0.5));
            acc += col.rgb * wt;
            wsum += wt;
        }

        // 第二轮：仍有洞就放宽边界（最多溢到画幅外 45%），做边缘外推补洞
        if (wsum <= 0.00001) {
            for (int i = 0; i < hp.count; ++i) {
                Img im = imgs[i];
                float3 c = toCamera(im, w);
                if (c.z >= -0.0001) { continue; }
                float inv = 1.0 / (-c.z);
                float tx = c.x * inv;
                float ty = c.y * inv;

                float over = max(fabs(tx) / im.ext.x, fabs(ty) / im.ext.y);
                if (over > 1.45) { continue; }
                float k = 1.0 - (max(over, 1.0) - 1.0) / 0.45;
                k = k * k;
                if (k <= 0.0001) { continue; }

                float su = clamp(0.5 + 0.5 * tx / im.misc.x, 0.0, 1.0);
                float sv = clamp(0.5 - 0.5 * ty / im.misc.y, 0.0, 1.0);

                float cosA2 = 1.0 / (1.0 + tx * tx + ty * ty);
                float wt = k * cosA2 * sqrt(cosA2) * im.misc.z * 0.35;

                float4 col = src.sample(smp, float2(su, sv), uint(im.misc.w + 0.5));
                acc += col.rgb * wt;
                wsum += wt;
            }
        }

        float3 outc;
        float filled;
        if (wsum > 0.00001) {
            outc = acc / wsum;
            filled = 1.0;
        } else {
            outc = float3(0.07, 0.08, 0.10);
            filled = 0.0;
        }
        dst.write(float4(outc, filled), gid);
    }
    """

    private struct Header {
        var outW: Int32 = 0
        var outH: Int32 = 0
        var count: Int32 = 0
        var pad: Int32 = 0
    }

    private struct ShaderImage {
        var r0 = SIMD4<Float>(0, 0, 0, 0)
        var r1 = SIMD4<Float>(0, 0, 0, 0)
        var r2 = SIMD4<Float>(0, 0, 0, 0)
        var misc = SIMD4<Float>(0, 0, 0, 0)
        var ext = SIMD4<Float>(0, 0, 0, 0)
    }

    /// 接缝羽化宽度（占画幅比例）。配准准了以后可以收窄，
    //    从 0.11 收到 0.07：过渡带窄了 1.6 倍，重影面积同比例缩小。
    private static let featherWidth: Float = 0.07

    // MARK: - 入口

    static func stitch(inputs rawInputs: [StitchInput],
                       outWidth: Int = 4096,
                       progress: @escaping (Double) -> Void) async throws -> StitchedPanorama {

        guard rawInputs.count >= 1 else { throw StitchError.noInput }
        guard let device = MTLCreateSystemDefaultDevice() else { throw StitchError.noDevice }

        // 着色器只能在 async 上下文编译；后面的重活全部丢到后台线程，
        // 否则拼一张 4096×2048 会把主线程卡住，进度条也不动。
        let library: MTLLibrary
        do {
            library = try await device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            throw StitchError.shader(error.localizedDescription)
        }
        guard let fn = library.makeFunction(name: "stitchPano") else {
            throw StitchError.shader("找不到 stitchPano")
        }
        let pipeline: MTLComputePipelineState
        do {
            pipeline = try await device.makeComputePipelineState(function: fn)
        } catch {
            throw StitchError.shader(error.localizedDescription)
        }

        return try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let out = try stitchSync(inputs: rawInputs,
                                             outWidth: outWidth,
                                             device: device,
                                             pipeline: pipeline,
                                             progress: progress)
                    cont.resume(returning: out)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - 主流程（后台线程）

    private static func stitchSync(inputs rawInputs: [StitchInput],
                                   outWidth: Int,
                                   device: MTLDevice,
                                   pipeline: MTLComputePipelineState,
                                   progress: @escaping (Double) -> Void) throws -> StitchedPanorama {

        let start = Date()

        // 1. 让第一张的位置落在等距圆柱图的中心（经度 0）
        let aligned = alignToFirst(rawInputs)
        guard !aligned.isEmpty else { throw StitchError.noInput }

        // 2. 光度精配准：修掉陀螺仪漂移，顺便解出逐帧增益与焦距偏差
        progress(0.04)
        var rotations = aligned.map(\.rotation)
        var gains = aligned.map { _ in Float(1) }
        var tans = aligned.map { (h: $0.tanHalfH, v: $0.tanHalfV) }
        var residualDeg: Double = 0
        var pairCount = 0

        if aligned.count >= 3 {
            var frames: [FrameAligner.Frame] = []
            frames.reserveCapacity(aligned.count)
            for input in aligned {
                guard let pyr = FrameAligner.pyramid(url: input.url) else { continue }
                frames.append(FrameAligner.Frame(rotation: input.rotation,
                                                 tanH: input.tanHalfH,
                                                 tanV: input.tanHalfV,
                                                 crop: input.crop,
                                                 pyramid: pyr))
            }
            if frames.count == aligned.count {
                let result = FrameAligner.refine(frames: frames, rounds: 4) { p in
                    progress(0.06 + 0.40 * p)
                }
                if let result {
                    rotations = result.rotations
                    gains = result.gains
                    residualDeg = Double(result.residualDeg)
                    pairCount = result.pairs
                    if abs(result.focalScale - 1) > 0.001 {
                        for k in 0..<tans.count {
                            tans[k].h = aligned[k].tanHalfH * result.focalScale
                            tans[k].v = aligned[k].tanHalfV * result.focalScale
                        }
                    }
                }
            }
        }

        progress(0.48)

        // 3. 输出尺寸（严格 2:1）
        let outW = max(1024, outWidth - (outWidth % 2))
        let outH = outW / 2

        // 4. 源图降采样尺寸：控制显存占用
        let count = aligned.count
        let budgetBytes = 170.0 * 1024 * 1024
        let approxBytesPerTexel = 4.0 / 0.75
        var maxDim = Int(sqrt(budgetBytes / (approxBytesPerTexel * Double(count))))
        maxDim = max(900, min(1600, maxDim))

        let firstSize = imagePixelSize(of: aligned[0].url)
        let aspect = firstSize.width > 0 ? firstSize.height / firstSize.width : 4.0 / 3.0
        let texW = maxDim
        let texH = max(2, Int((Double(maxDim) * Double(aspect)).rounded()))

        // 5. 纹理数组
        let desc = MTLTextureDescriptor()
        desc.textureType = .type2DArray
        desc.pixelFormat = .rgba8Unorm
        desc.width = texW
        desc.height = texH
        desc.arrayLength = count
        desc.mipmapLevelCount = 1
        desc.usage = .shaderRead
        desc.storageMode = .shared
        guard let srcTex = device.makeTexture(descriptor: desc) else {
            throw StitchError.texture("源纹理分配失败（\(texW)x\(texH)x\(count)）")
        }

        // 6. 参数表
        var params = [ShaderImage]()
        params.reserveCapacity(count)
        var slice = 0

        for (i, input) in aligned.enumerated() {
            guard let bytes = downsampledRGBA(url: input.url, width: texW, height: texH) else { continue }

            bytes.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    srcTex.replace(region: MTLRegionMake2D(0, 0, texW, texH),
                                   mipmapLevel: 0,
                                   slice: slice,
                                   withBytes: base,
                                   bytesPerRow: texW * 4,
                                   bytesPerImage: texW * 4 * texH)
                }
            }

            let crop = clampf(input.crop, 0.6, 1.0)
            let tanFullH = max(0.05, tans[i].h)
            let tanFullV = max(0.05, tans[i].v)

            var p = ShaderImage()
            let rows = rotations[i].worldToCameraRows4
            p.r0 = rows[0]
            p.r1 = rows[1]
            p.r2 = rows[2]
            p.misc = SIMD4<Float>(tanFullH, tanFullV, gains[i], Float(slice))
            p.ext = SIMD4<Float>(tanFullH * crop, tanFullV * crop, 0.5 * crop, featherWidth)
            params.append(p)
            slice += 1

            progress(0.48 + 0.34 * Double(i + 1) / Double(count))
        }

        guard !params.isEmpty else { throw StitchError.noInput }

        // 7. 输出纹理
        let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                               width: outW,
                                                               height: outH,
                                                               mipmapped: false)
        outDesc.usage = [.shaderWrite, .shaderRead]
        outDesc.storageMode = .shared
        guard let outTex = device.makeTexture(descriptor: outDesc) else {
            throw StitchError.texture("输出纹理分配失败")
        }

        // 8. 执行
        guard let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else {
            throw StitchError.texture("无法创建命令队列")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(srcTex, index: 0)
        encoder.setTexture(outTex, index: 1)

        var header = Header()
        header.outW = Int32(outW)
        header.outH = Int32(outH)
        header.count = Int32(params.count)
        encoder.setBytes(&header, length: MemoryLayout<Header>.stride, index: 0)

        // 参数表用 buffer 而不是 setBytes：setBytes 有 4KB 上限，机位多时会超
        let paramLength = MemoryLayout<ShaderImage>.stride * params.count
        let paramBuffer: MTLBuffer? = params.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: paramLength, options: .storageModeShared)
        }
        guard let paramBuffer else { throw StitchError.texture("参数缓冲分配失败") }
        encoder.setBuffer(paramBuffer, offset: 0, index: 1)

        let tw = pipeline.threadExecutionWidth
        let th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
        encoder.dispatchThreads(MTLSize(width: outW, height: outH, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        if let err = cmd.error {
            throw StitchError.texture(err.localizedDescription)
        }

        progress(0.86)

        // 9. 读回并统计空洞
        var bytes = [UInt8](repeating: 0, count: outW * outH * 4)
        bytes.withUnsafeMutableBytes { raw in
            if let base = raw.baseAddress {
                outTex.getBytes(base,
                                bytesPerRow: outW * 4,
                                bytesPerImage: outW * 4 * outH,
                                from: MTLRegionMake2D(0, 0, outW, outH),
                                mipmapLevel: 0,
                                slice: 0)
            }
        }
        var holes = 0
        var idx = 3
        let total = outW * outH
        while idx < bytes.count {
            if bytes[idx] == 0 { holes += 1 }
            idx += 4
        }
        let coverage = total > 0 ? Double(total - holes) / Double(total) : 0

        progress(0.94)

        // 10. 生成 CGImage
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cg = CGImage(width: outW,
                               height: outH,
                               bitsPerComponent: 8,
                               bitsPerPixel: 32,
                               bytesPerRow: outW * 4,
                               space: colorSpace,
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                               provider: provider,
                               decode: nil,
                               shouldInterpolate: true,
                               intent: .defaultIntent) else {
            throw StitchError.encode
        }

        guard let jpeg = UIImage(cgImage: cg, scale: 1, orientation: .up)
            .jpegData(compressionQuality: 0.92) else {
            throw StitchError.encode
        }

        let thumbnail = makeThumbnail(from: cg)
        progress(1.0)

        return StitchedPanorama(equirect: cg,
                                jpeg: jpeg,
                                thumbnail: thumbnail,
                                coverage: coverage,
                                seconds: Date().timeIntervalSince(start),
                                width: outW,
                                height: outH,
                                residualDeg: residualDeg,
                                pairCount: pairCount)
    }

    // MARK: - 辅助

    /// 把所有姿态整体旋转，使第一张画面位于经度 0（等距圆柱图的中间）
    private static func alignToFirst(_ inputs: [StitchInput]) -> [StitchInput] {
        guard let first = inputs.first else { return inputs }
        let f = first.rotation.cameraForward
        let a = rotationZ(-yawOf(f))
        return inputs.map { input in
            StitchInput(url: input.url,
                        rotation: simd_mul(a, input.rotation),
                        tanHalfH: input.tanHalfH,
                        tanHalfV: input.tanHalfV,
                        crop: input.crop,
                        luma: input.luma)
        }
    }

    private static func imagePixelSize(of url: URL) -> CGSize {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Double,
              let h = props[kCGImagePropertyPixelHeight] as? Double else {
            return .zero
        }
        return CGSize(width: w, height: h)
    }

    /// 按目标尺寸解码并重绘为 RGBA8（避免整张全分辨率图驻留内存）
    private static func downsampledRGBA(url: URL, width: Int, height: Int) -> [UInt8]? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height)
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }

        var data = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let ctx = CGContext(data: base,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width * 4,
                                      space: colorSpace,
                                      bitmapInfo: info) else { return }
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return data
    }

    /// 从等距圆柱图中央裁一块正常视角的画面做封面
    private static func makeThumbnail(from cg: CGImage) -> Data {
        let w = CGFloat(cg.width)
        let h = CGFloat(cg.height)
        let cropW = min(w, 900)
        let cropH = min(h, cropW * 0.72)
        let rect = CGRect(x: (w - cropW) / 2, y: (h - cropH) / 2, width: cropW, height: cropH)
        let cropped = cg.cropping(to: rect) ?? cg
        let target = UIImage(cgImage: cropped, scale: 1, orientation: .up)
        return target.jpegData(compressionQuality: 0.85) ?? Data()
    }
}
