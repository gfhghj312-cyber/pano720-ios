"""生成 App 图标：深空渐变 + 球面线框（与 App 内 SphereMap 视觉语言一致）"""
from PIL import Image, ImageDraw, ImageFilter
import math

S = 2048          # 超采样尺寸
OUT = 1024

def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

img = Image.new("RGB", (S, S), (7, 14, 26))
d = ImageDraw.Draw(img)

# 斜向渐变背景
top = (10, 16, 34)
bot = (26, 18, 62)
for y in range(S):
    t = y / (S - 1)
    d.line([(0, y), (S, y)], fill=lerp(top, bot, t ** 0.85))

# 顶部光晕
glow = Image.new("RGB", (S, S), (0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.ellipse([S * 0.12, -S * 0.35, S * 0.88, S * 0.42], fill=(30, 90, 150))
glow = glow.filter(ImageFilter.GaussianBlur(S // 9))
img = Image.blend(img, Image.blend(img, glow, 0.55), 0.55)
d = ImageDraw.Draw(img)

cx = cy = S / 2
R = S * 0.325

MINT = (102, 232, 196)
SKY = (92, 200, 255)
VIOL = (150, 140, 255)

# 球体线框
def ellipse_ring(rx, ry, color, width):
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=color, width=width)

w = int(S * 0.0075)
ellipse_ring(R, R, SKY + (255,), int(w * 1.6))

lat = [0.42, 0.70, 0.90]
for k in lat:
    ellipse_ring(R, R * k, SKY + (150,), w)

n = 6
for i in range(n):
    ang = math.pi * i / n
    rx = abs(math.cos(ang)) * R
    ellipse_ring(max(rx, R * 0.03), R, SKY + (110,), w)

# 赤道上的拍摄进度弧
d.arc([cx - R, cy - R, cx + R, cy + R], start=-160, end=25, fill=MINT, width=int(w * 3.4))
# 弧线端点
for a in (-160, 25):
    px = cx + math.cos(math.radians(a)) * R
    py = cy + math.sin(math.radians(a)) * R
    rr = w * 3.2
    d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=MINT)

# 中心光圈（相机）
ap_r = S * 0.072
for i in range(3):
    t = i / 3
    col = lerp(MINT, VIOL, t)
    rr = ap_r * (1 - t * 0.55)
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
              outline=col + (220,), width=int(w * 2.2))
d.ellipse([cx - ap_r * 0.22, cy - ap_r * 0.22, cx + ap_r * 0.22, cy + ap_r * 0.22],
          fill=(255, 255, 255))

# 光晕叠加
halo = Image.new("RGB", (S, S), (0, 0, 0))
hd = ImageDraw.Draw(halo)
hd.ellipse([cx - R * 1.15, cy - R * 1.15, cx + R * 1.15, cy + R * 1.15],
           outline=(60, 150, 220), width=int(S * 0.05))
halo = halo.filter(ImageFilter.GaussianBlur(S // 22))
img = Image.blend(img, Image.blend(img, halo, 0.30), 0.5)

img = img.resize((OUT, OUT), Image.LANCZOS)
img.save("Resources/Assets.xcassets/AppIcon.appiconset/icon-1024.png", "PNG")
print("icon written", img.size)
