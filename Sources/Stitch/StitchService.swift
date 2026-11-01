import Foundation
import UIKit
import Metal
import CoreGraphics
import ImageIO
import simd

struct StitchInput {
    let url: URL
    let rotation: simd_float3x3   // 设备(=相机) -> 世界
    let tanHalfH: Float
    let tanHalfV: Float
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

/// 基于姿态（陀螺仪）的球面投影拼接：
/// 对等距圆柱目标图的每个像素求世界射线，反投影到每一张源图采样，
/// 用边缘羽化 + 视角余弦权重做加权混合，并做亮度增益补偿。
enum StitchService {

    // MARK: - Metal 着色器（运行时编译，免去 .metal 构建配置）

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Header {
        int outW;
        int outH;
        int count;
        float pad;
    };

    struct Img {
        float4 r0;
        float4 r1;
        float4 r2;
        float4 misc;   // x: tanHalfH, y: tanHalfV, z: gain, w: slice
    };

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
        float3 wdir = float3(cl * cos(lon), cl * sin(lon), sin(lat));

        float3 acc = float3(0.0);
        float wsum = 0.0;

        for (int i = 0; i < hp.count; ++i) {
            Img im = imgs[i];
            float3 c = float3(dot(im.r0.xyz, wdir),
                              dot(im.r1.xyz, wdir),
                              dot(im.r2.xyz, wdir));
            if (c.z >= -0.0001) { continue; }

            float iz = 1.0 / (-c.z);
            float tx = c.x * iz;
            float ty = c.y * iz;
            float su = 0.5 + 0.5 * tx / im.misc.x;
            float sv = 0.5 - 0.5 * ty / im.misc.y;
            if (su < 0.0 || su > 1.0 || sv < 0.0 || sv > 1.0) { continue; }

            float eu = min(su, 1.0 - su);
            float ev = min(sv, 1.0 - sv);
            float feather = smoothstep(0.0, 0.17, eu) * smoothstep(0.0, 0.17, ev);
            if (feather <= 0.0001) { continue; }

            // 越靠近光轴权重越高，减少边缘畸变带来的错位
            float cosA = 1.0 / sqrt(1.0 + tx * tx + ty * ty);
            float wt = feather * cosA * cosA * cosA * im.misc.z;

            float4 col = src.sample(smp, float2(su, sv), uint(im.misc.w + 0.5));
            acc += col.rgb * wt;
            wsum += wt;
        }

        float3 outc;
        float filled;
        if (wsum > 1e-5) {
            outc = acc / max(wsum, 1e-5);
            filled = 1.0;
        } else {
            outc = float3(0.10, 0.11, 0.13);
            filled = 0.0;
        }
        dst.write(float4(outc, filled), gid);
    }
    """

    private struct Header {
        var outW: Int32 = 0
        var outH: Int32 = 0
        var count: Int32 = 0
        var pad: Float = 0
    }

    private struct ShaderImage {
        var r0 = SIMD4<Float>(0, 0, 0, 0)
        var r1 = SIMD4<Float>(0, 0, 0, 0)
        var r2 = SIMD4<Float>(0, 0, 0, 0)
        var misc = SIMD4<Float>(0, 0, 0, 0)
    }

    // MARK: - 入口

    static func stitch(inputs rawInputs: [StitchInput],
                       outWidth: Int = 4096,
                       progress: @escaping (Double) -> Void) async throws -> StitchedPanorama {

        guard !rawInputs.isEmpty else { throw StitchError.noInput }
        guard let device = MTLCreateSystemDefaultDevice() else { throw StitchError.noDevice }

        let start = Date()

        // 1. 让第一张的位置落在等距圆柱图的中心（经度 0），接缝留在正后方
        let inputs = alignToFirst(rawInputs)

        // 2. 输出尺寸（严格 2:1）
        let outW = max(1024, outWidth)
        let outH = outW / 2

        // 3. 源图降采样尺寸：控制显存占用
        let count = inputs.count
        let budgetBytes = 140.0 * 1024 * 1024
        let approxBytesPerTexel = 4.0 / 0.75   // 4:3 情况下每"最大边长"像素的字节数近似
        var maxDim = Int(sqrt(budgetBytes / (approxBytesPerTexel * Double(count))))
        maxDim = max(900, min(1600, maxDim))

        let firstSize = imagePixelSize(of: inputs[0].url)
        let aspect = firstSize.width > 0 ? firstSize.height / firstSize.width : 4.0 / 3.0
        let texW = maxDim
        let texH = max(2, Int((Double(maxDim) * Double(aspect)).rounded()))

        // 4. 管线
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

        // 6. 亮度增益（全局均值回归，抑制自动曝光跳变）
        let lumas = inputs.map { max(0.02, $0.luma) }
        let globalLuma = lumas.reduce(0, +) / Float(lumas.count)

        var params = [ShaderImage]()
        params.reserveCapacity(count)

        for (i, input) in inputs.enumerated() {
            let gain = clampf(globalLuma / lumas[i], 0.72, 1.38)

            guard let bytes = downsampledRGBA(url: input.url, width: texW, height: texH) else { continue }
            bytes.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    srcTex.replace(region: MTLRegionMake2D(0, 0, texW, texH),
                                   slice: i,
                                   mipmapLevel: 0,
                                   withBytes: base,
                                   bytesPerRow: texW * 4)
                }
            }

            let rows = input.rotation.rows4
            var p = ShaderImage()
            p.r0 = rows[0]
            p.r1 = rows[1]
            p.r2 = rows[2]
            p.misc = SIMD4<Float>(input.tanHalfH, input.tanHalfV, gain, Float(i))
            params.append(p)

            progress(0.05 + 0.65 * Double(i + 1) / Double(count))
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
        params.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                encoder.setBytes(base, length: MemoryLayout<ShaderImage>.stride * params.count, index: 1)
            }
        }

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

        progress(0.85)

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
                                height: outH)
    }

    // MARK: - 辅助

    /// 把所有姿态整体旋转，使第一张画面位于经度 0（等距圆柱图的中间）
    private static func alignToFirst(_ inputs: [StitchInput]) -> [StitchInput] {
        guard let first = inputs.first else { return inputs }
        let f = first.rotation.cameraForward
        let yaw0 = yawOf(f)
        let a = rotationZ(-yaw0)
        return inputs.map { input in
            StitchInput(url: input.url,
                        rotation: simd_mul(a, input.rotation),
                        tanHalfH: input.tanHalfH,
                        tanHalfV: input.tanHalfV,
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
