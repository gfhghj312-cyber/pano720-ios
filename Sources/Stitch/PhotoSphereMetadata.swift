import Foundation

/// 把 Photo Sphere / GPano XMP 元数据注入 JPEG，让系统「照片」、Google Photos、
/// Facebook 等把这张 2:1 等距圆柱图识别成可拖动的 360° 全景。
///
/// 字段来自 Google Street View 的 `libphotosphere_utils.so` 与
/// `cityblock/portable/panorama/session/photosphere_parameters.cc`
/// 中实际使用的 `geo_photo_service.PanoramaParameters`：
///   FullPanoWidthPixels / FullPanoHeightPixels / CroppedAreaImage*Pixels /
///   CroppedAreaTopPixels / CroppedAreaLeftPixels / Pose*Degrees / InitialView*Degrees
/// 命名空间 `http://ns.google.com/photos/1.0/panorama/` 是 Google 公开的
/// Photo Sphere XMP 规范，属于跨平台数据格式，不依赖 Android 运行时。
enum PhotoSphereMetadata {

    /// - Parameters:
    ///   - jpeg: 已编码的 JPEG
    ///   - width/height: 成片像素尺寸（完整球面，无裁切）
    ///   - headingDegrees: 拍摄起始方向的磁航向，可选
    static func inject(into jpeg: Data,
                       width: Int,
                       height: Int,
                       headingDegrees: Double = 0) -> Data {
        let xmp = packet(width: width, height: height,
                         heading: headingDegrees).data(using: .utf8) ?? Data()

        // XMP APP1 的标准标识头（28 字符 + 1 个 NUL）
        var payload = Data("http://ns.adobe.com/xap/1.0/".utf8)
        payload.append(0)
        payload.append(xmp)

        let segmentLength = 2 + payload.count
        guard segmentLength <= 0xFFFF else { return jpeg }

        var segment = Data([
            0xFF, 0xE1,
            UInt8(truncatingIfNeeded: segmentLength >> 8),
            UInt8(truncatingIfNeeded: segmentLength)
        ])
        segment.append(payload)

        let bytes = [UInt8](jpeg)
        guard bytes.count > 4, bytes[0] == 0xFF, bytes[1] == 0xD8 else { return jpeg }

        // 插在 SOI 与已有 APP0 / APP1(Exif) 之后、第一个图像段之前。
        // 不要插到 SOS (0xDA) 之后，否则会破坏压缩数据。
        var offset = 2
        while offset + 4 <= bytes.count {
            guard bytes[offset] == 0xFF else { break }
            let marker = bytes[offset + 1]
            if marker == 0xD9 || marker == 0xDA || marker == 0x00 { break }
            if marker != 0xE0 && marker != 0xE1 { break }
            let length = (Int(bytes[offset + 2]) << 8) | Int(bytes[offset + 3])
            if length < 2 { break }
            offset += 2 + length
        }
        var out = Data(bytes[0..<offset])
        out.append(segment)
        out.append(Data(bytes[offset..<bytes.count]))
        return out
    }

    private static func packet(width: Int, height: Int, heading: Double) -> String {
        let h = String(format: "%.2f", heading)
        return """
        <?xpacket begin="\u{FEFF}" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        <rdf:Description rdf:about=""
          xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"
          GPano:UsePanoramaViewer="True"
          GPano:ProjectionType="equirectangular"
          GPano:FullPanoWidthPixels="\(width)"
          GPano:FullPanoHeightPixels="\(height)"
          GPano:CroppedAreaImageWidthPixels="\(width)"
          GPano:CroppedAreaImageHeightPixels="\(height)"
          GPano:CroppedAreaLeftPixels="0"
          GPano:CroppedAreaTopPixels="0"
          GPano:PoseHeadingDegrees="\(h)"
          GPano:PosePitchDegrees="0.0"
          GPano:PoseRollDegrees="0.0"
          GPano:InitialViewHeadingDegrees="\(h)"
          GPano:InitialViewPitchDegrees="0.0"
          GPano:InitialViewRollDegrees="0.0"/>
        </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end="w"?>
        """
    }
}
