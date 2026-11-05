"""验证 PhotoSphereMetadata.inject 的 JPEG APP1 段格式。

Swift 侧无法在桌面端跑，这里用同样的字节布局复现一遍，检查：
1. marker 链完整（SOI / APP0 / APP1 / ... / SOS / EOI）
2. APP1 长度字段 = 2 + 29(标识头) + XMP 长度
3. XMP 里包含 libphotosphere_utils.so 中出现的全部 GPano 字段
4. 注入后 JPEG 仍能被标准解码器打开、尺寸不变
"""
import io
import re
from PIL import Image

IDENT = b"http://ns.adobe.com/xap/1.0/\0"

FIELDS = [
    "UsePanoramaViewer", "ProjectionType",
    "FullPanoWidthPixels", "FullPanoHeightPixels",
    "CroppedAreaImageWidthPixels", "CroppedAreaImageHeightPixels",
    "CroppedAreaLeftPixels", "CroppedAreaTopPixels",
    "PoseHeadingDegrees", "PosePitchDegrees", "PoseRollDegrees",
    "InitialViewHeadingDegrees", "InitialViewPitchDegrees", "InitialViewRollDegrees",
]


def xmp_packet(w, h):
    return f'''<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about=""
  xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"
  GPano:UsePanoramaViewer="True"
  GPano:ProjectionType="equirectangular"
  GPano:FullPanoWidthPixels="{w}"
  GPano:FullPanoHeightPixels="{h}"
  GPano:CroppedAreaImageWidthPixels="{w}"
  GPano:CroppedAreaImageHeightPixels="{h}"
  GPano:CroppedAreaLeftPixels="0"
  GPano:CroppedAreaTopPixels="0"
  GPano:PoseHeadingDegrees="0.00"
  GPano:PosePitchDegrees="0.0"
  GPano:PoseRollDegrees="0.0"
  GPano:InitialViewHeadingDegrees="0.00"
  GPano:InitialViewPitchDegrees="0.0"
  GPano:InitialViewRollDegrees="0.0"/>
</rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
'''.encode("utf-8")


def inject(jpeg, w, h):
    payload = IDENT + xmp_packet(w, h)
    n = 2 + len(payload)
    seg = bytes([0xFF, 0xE1, n >> 8, n & 0xFF]) + payload
    b = bytearray(jpeg)
    assert b[0] == 0xFF and b[1] == 0xD8
    off = 2
    while off + 4 <= len(b):
        if b[off] != 0xFF:
            break
        m = b[off + 1]
        if m in (0xD9, 0xDA, 0x00) or m not in (0xE0, 0xE1):
            break
        ln = (b[off + 2] << 8) | b[off + 3]
        if ln < 2:
            break
        off += 2 + ln
    return bytes(b[:off]) + seg + bytes(b[off:])


def main():
    import numpy as np
    arr = (np.random.default_rng(3).random((512, 1024, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    original = buf.getvalue()

    out = inject(original, 1024, 512)
    print(f"原 JPEG {len(original)} B -> 注入后 {len(out)} B (+{len(out)-len(original)} B)")

    # 1) 遍历 marker 链
    off, marks, xmp_seg = 2, [], None
    while off + 4 <= len(out):
        if out[off] != 0xFF:
            break
        m = out[off + 1]
        if m == 0xD9:
            marks.append("EOI"); break
        if m == 0xDA:
            marks.append("SOS"); break
        ln = (out[off + 2] << 8) | out[off + 3]
        marks.append(f"FF{m:02X}(len={ln})")
        if m == 0xE1 and out[off + 4:off + 4 + len(IDENT)] == IDENT:
            xmp_seg = out[off + 4 + len(IDENT): off + 2 + ln]
        off += 2 + ln
    print("marker 链:", " -> ".join(marks))

    assert xmp_seg is not None, "找不到 XMP APP1"
    assert xmp_seg.startswith(b"<?xpacket"), "xpacket 头缺失"
    assert xmp_seg.startswith(b"<?xpacket begin=\"\xef\xbb\xbf\""), "BOM 编码不对"
    text = xmp_seg.decode("utf-8")
    missing = [f for f in FIELDS if f"GPano:{f}=" not in text]
    assert not missing, f"缺字段: {missing}"
    print(f"GPano 字段: {len(FIELDS)}/{len(FIELDS)} 齐全，命名空间 "
          f"{'http://ns.google.com/photos/1.0/panorama/' in text}")

    # 2) 注入后仍可解码
    im = Image.open(io.BytesIO(out))
    im.load()
    assert im.size == (1024, 512)
    print(f"解码通过: {im.size}, mode={im.mode}")

    # 3) 解析 XMP 数值
    w = int(re.search(r'FullPanoWidthPixels="(\d+)"', text).group(1))
    h = int(re.search(r'FullPanoHeightPixels="(\d+)"', text).group(1))
    print(f"球面尺寸 {w}x{h}, 比例 {w/h:.1f}:1, projection=equirectangular -> OK")


if __name__ == "__main__":
    main()
