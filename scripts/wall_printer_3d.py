"""Render a wall printer printing an image, filmed from 45 degrees to the side.

A small 3D scene (wall, wooden floor, floor rail and a vertical wall printer) is
projected through a pinhole camera that looks at the wall from 45 degrees off to
the left. The wall and floor are textured planes warped with homographies; the
printer is built from shaded boxes. The print head runs up and down its mast,
laying the image down one vertical swath at a time while the machine rolls left
to right along the rail. The image only appears where the head has passed.

Usage:
    python scripts/wall_printer_3d.py IMAGE OUTPUT.mp4 [--fps 30] [--angle 45]

Requires Pillow, numpy and ffmpeg on PATH.
"""
import argparse
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
SS = 2  # supersampling for the printer layer

# Scene, in centimetres. The wall is the plane z=0, the floor is y=0 and the
# room (and the camera) is on the negative z side.
WALL_X = (-150, 1400)
WALL_Y = (0, 420)
WALL_PPC = 5  # texture pixels per cm
MURAL = (60, 40, 340, 196)  # x0, y0, x1, y1 on the wall
FLOOR_X = (-150, 1400)
FLOOR_Z = (-260, 0)
FLOOR_PPC = 3
SKIRTING = 10

SWATH = 14  # cm printed per vertical pass
PASS_FRAMES = 18
STEP_FRAMES = 6
INTRO_FRAMES = 50
EXIT_FRAMES = 60
HOLD_FRAMES = 90

RAIL_Z = (-52, -40)
MAST_TOP = 212
LIGHT = np.array([-0.55, 0.75, -0.65])
LIGHT /= np.linalg.norm(LIGHT)


class Camera:
    def __init__(self, angle_deg, distance, height, target):
        a = np.radians(angle_deg)
        self.target = np.array(target, float)
        self.pos = self.target + distance * np.array([-np.sin(a), 0, -np.cos(a)])
        self.pos[1] = height
        f = self.target - self.pos
        self.f = f / np.linalg.norm(f)
        r = np.cross([0, 1, 0], self.f)
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.f, self.r)
        self.focal, self.cx, self.cy = 1.0, 0.0, 0.0

    def depth(self, p):
        return (np.asarray(p, float) - self.pos) @ self.f

    def project(self, p):
        d = np.asarray(p, float) - self.pos
        z = d @ self.f
        x = d @ self.r / z
        y = -(d @ self.u) / z
        return np.stack([self.cx + self.focal * x, self.cy + self.focal * y], -1)

    def fit(self, pts, margin):
        """Choose focal length and principal point so pts fill the frame."""
        self.focal, self.cx, self.cy = 1.0, 0.0, 0.0
        p = self.project(pts)
        lo, hi = p.min(0), p.max(0)
        s = min(W * (1 - 2 * margin) / (hi - lo)[0], H * (1 - 2 * margin) / (hi - lo)[1])
        self.focal = s
        self.cx = W / 2 - s * (lo[0] + hi[0]) / 2
        self.cy = H / 2 - s * (lo[1] + hi[1]) / 2


def homography(src, dst):
    """Coefficients for PIL's PERSPECTIVE transform mapping output dst -> input src."""
    a = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    return np.linalg.solve(np.array(a, float), np.array(src, float).reshape(8))


def warp(tex, cam, corners3d, resample=Image.BICUBIC):
    """Warp a texture whose corners (tl, tr, br, bl) sit at corners3d in the scene."""
    tw, th = tex.size
    src = [(0, 0), (tw, 0), (tw, th), (0, th)]
    dst = cam.project(np.array(corners3d, float))
    return tex.transform((W, H), Image.PERSPECTIVE, tuple(homography(src, dst)), resample)


def wall_texture(wall_rgb):
    tw = (WALL_X[1] - WALL_X[0]) * WALL_PPC
    th = (WALL_Y[1] - WALL_Y[0]) * WALL_PPC
    rng = np.random.default_rng(3)
    tex = np.empty((th, tw, 3), np.float32)
    tex[:] = wall_rgb
    noise = rng.normal(0, 3.0, (th // 3, tw // 3)).astype(np.float32)
    noise = np.asarray(Image.fromarray(noise).resize((tw, th), Image.BICUBIC))
    xs = np.linspace(*WALL_X, tw)
    ys = np.linspace(WALL_Y[1], WALL_Y[0], th)
    # Daylight from the left, falling off along the wall and towards the floor.
    light = 1.08 - 0.22 * np.clip((xs - WALL_X[0]) / 700, 0, 1)
    light = light[None, :] * (1.0 - 0.06 * np.clip((200 - ys) / 200, 0, 1))[:, None]
    # Ambient occlusion where the wall meets the floor.
    light *= (1 - 0.18 * np.exp(-ys / 6))[:, None]
    tex = tex * light[..., None] + noise[..., None]
    sk = ys < SKIRTING
    tex[sk] = np.array([238, 235, 228], np.float32) * light[sk][..., None] * 0.98
    tex[(ys >= SKIRTING) & (ys < SKIRTING + 0.6)] *= 0.8
    return np.clip(tex, 0, 255).astype(np.uint8), xs, ys


def mural_texture(path, wall, xs, ys):
    full = wall.copy()
    x0, y0, x1, y1 = MURAL
    c0, c1 = np.searchsorted(xs, x0), np.searchsorted(xs, x1)
    r0, r1 = np.searchsorted(-ys, -y1), np.searchsorted(-ys, -y0)
    ink = np.asarray(Image.open(path).convert("RGB").resize((c1 - c0, r1 - r0), Image.LANCZOS), np.float32)
    bg = np.concatenate([ink[:8].reshape(-1, 3), ink[-8:].reshape(-1, 3)]).mean(0)
    alpha = np.clip((np.abs(ink - bg).sum(-1) - 35) / 40, 0, 1)
    # Keep only the window itself (and its cast shadow just around it) so the
    # image's own plaster never shows as a lighter rectangle on the wall.
    rows = np.where((alpha > 0.5).mean(1) > 0.3)[0]
    cols = np.where((alpha > 0.5).mean(0) > 0.3)[0]
    pad = int(0.02 * alpha.shape[1])
    keep = np.zeros_like(alpha)
    keep[max(rows[0] - pad, 0):rows[-1] + pad, max(cols[0] - pad, 0):cols[-1] + pad] = 1
    keep = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(pad / 2)), np.float32) / 255
    alpha *= keep
    alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    under = wall[r0:r1, c0:c1].astype(np.float32)
    printed = ink * under / bg
    full[r0:r1, c0:c1] = np.clip(under * (1 - alpha[..., None]) + printed * alpha[..., None], 0, 255)
    return full, (r0, r1, c0, c1)


def floor_texture():
    tw = (FLOOR_X[1] - FLOOR_X[0]) * FLOOR_PPC
    th = (FLOOR_Z[1] - FLOOR_Z[0]) * FLOOR_PPC
    rng = np.random.default_rng(11)
    tex = np.zeros((th, tw, 3), np.float32)
    plank_h = 18 * FLOOR_PPC
    base = np.array([138, 102, 74], np.float32)
    for i, r in enumerate(range(0, th, plank_h)):
        offset = rng.integers(0, 120 * FLOOR_PPC)
        for c in range(-offset, tw, 120 * FLOOR_PPC):
            tone = base * rng.uniform(0.88, 1.06) + rng.normal(0, 2, 3)
            tex[r:r + plank_h, max(c, 0):c + 120 * FLOOR_PPC] = tone
            tex[r:r + plank_h, max(c, 0):max(c, 0) + 2] *= 0.7
        tex[r:r + 2] *= 0.65
    grain = rng.normal(0, 1, (th, tw // 40)).astype(np.float32)
    grain = np.asarray(Image.fromarray(grain).resize((tw, th), Image.BICUBIC)) * 5
    tex += grain[..., None]
    # Row 0 is the far edge at the wall (z=0): darken it for contact shadow.
    zs = np.linspace(FLOOR_Z[1], FLOOR_Z[0], th)
    tex *= (1 - 0.3 * np.exp(zs / 8))[:, None, None]
    return Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8))


def box(x0, x1, y0, y1, z0, z1, color, lit=True):
    return dict(lo=np.array([x0, y0, z0], float), hi=np.array([x1, y1, z1], float), color=np.array(color, float), lit=lit)


FACES = [  # (normal, corner indices) with corners indexed by (ix, iy, iz) bits
    ((-1, 0, 0), [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
    ((1, 0, 0), [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]),
    ((0, -1, 0), [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]),
    ((0, 1, 0), [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
    ((0, 0, -1), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
    ((0, 0, 1), [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]),
]


def printer_boxes(cx, hy):
    hw = SWATH / 2
    rz0, rz1 = RAIL_Z
    bz0, bz1 = rz0 - 8, rz1 + 4
    boxes = [
        # Wheels and base chassis sitting on the rail.
        *[box(wx - 3, wx + 3, 0, 7, wz - 3, wz + 3, (25, 25, 28)) for wx in (cx - 26, cx + 26) for wz in (bz0 + 4, bz1 - 4)],
        box(cx - 32, cx + 32, 6, 20, bz0, bz1, (44, 47, 55)),
        box(cx - 32.5, cx + 32.5, 13, 15, bz0 - 0.5, bz0 + 1, (66, 165, 245), lit=False),
        # Control box with a small screen.
        box(cx + 10, cx + 28, 20, 34, bz0 + 2, bz0 + 12, (52, 56, 64)),
        box(cx + 13, cx + 25, 24, 31, bz0 + 1.4, bz0 + 2, (40, 150, 220), lit=False),
        # Aluminium mast and top cap.
        box(cx - 4, cx + 4, 20, MAST_TOP, -36, -28, (196, 199, 204)),
        box(cx - 5, cx + 5, MAST_TOP, MAST_TOP + 4, -37, -27, (55, 58, 64)),
        # Carriage riding the mast and the print head reaching towards the wall.
        box(cx - 6, cx + 6, hy - 9, hy + 9, -38, -27, (60, 64, 72)),
        box(cx - hw - 3, cx + hw + 3, hy - 5, hy + 5, -27, -3, (34, 37, 44)),
        box(cx + hw - 1, cx + hw + 1, hy + 1, hy + 3, -14, -12, (80, 255, 140), lit=False),
    ]
    for i, c in enumerate([(0, 174, 239), (236, 0, 140), (255, 225, 0), (25, 25, 25)]):
        x = cx - hw + 1 + i * 3.2
        boxes.append(box(x, x + 2.4, hy + 5, hy + 8, -22, -16, c))
    return boxes


def rail_boxes():
    rz0, rz1 = RAIL_Z
    return [box(-140, 700, 0, 2.5, rz0, rz1, (176, 180, 186))]


def shade(color, normal, lit):
    if not lit:
        return color
    k = 0.42 + 0.62 * max(0.0, float(np.dot(normal, LIGHT)))
    return np.clip(color * k, 0, 255)


def draw_boxes(draw, cam, boxes, scale):
    faces = []
    for b in boxes:
        lo, hi = b["lo"], b["hi"]
        for n, idx in FACES:
            pts = np.array([[(hi if i else lo)[0], (hi if j else lo)[1], (hi if k else lo)[2]] for i, j, k in idx])
            centre = pts.mean(0)
            if np.dot(n, cam.pos - centre) <= 0:
                continue  # back face
            faces.append((cam.depth(centre), pts, shade(b["color"], np.array(n, float), b["lit"])))
    faces.sort(key=lambda f: -f[0])
    for _, pts, col in faces:
        p = cam.project(pts) * scale
        draw.polygon([tuple(q) for q in p], fill=tuple(int(c) for c in col) + (255,))


def convex_hull(points):
    pts = sorted(map(tuple, points))
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def shadow_layer(cam, boxes):
    """Soft shadow of the printer cast onto the wall and floor."""
    layer = Image.new("L", (W // 2, H // 2), 0)
    d = ImageDraw.Draw(layer)
    for b in boxes:
        lo, hi = b["lo"], b["hi"]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        hits = []
        for p in corners:
            t_wall = -p[2] / -LIGHT[2]
            q = p - LIGHT * t_wall
            if q[1] < 0:  # reaches the floor before the wall
                q = p - LIGHT * (p[1] / LIGHT[1])
            hits.append(q)
        pts = cam.project(np.array(hits)) / 2
        hull = convex_hull(pts)
        if len(hull) >= 3:
            d.polygon(hull, fill=95)
    return layer.filter(ImageFilter.GaussianBlur(9)).resize((W, H), Image.BILINEAR)


def ease(t):
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("output")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--angle", type=float, default=45, help="camera yaw from straight-on, degrees")
    ap.add_argument("--distance", type=float, default=750, help="camera distance from the mural centre, cm")
    ap.add_argument("--stills", help="comma-separated frame numbers to save as JPEGs instead of a video")
    args = ap.parse_args()

    src = Image.open(args.image).convert("RGB")
    wall_rgb = np.asarray(src.crop((5, 5, 60, 60)), np.float32).reshape(-1, 3).mean(0)
    wall, xs, ys = wall_texture(wall_rgb)
    full, (r0, r1, c0, c1) = mural_texture(args.image, wall, xs, ys)

    mx0, my0, mx1, my1 = MURAL
    cam = Camera(args.angle, distance=args.distance, height=135, target=((mx0 + mx1) / 2, 115, 0))
    fit_pts = np.array([[mx0 - 15, 8, -20], [mx0 - 15, MAST_TOP + 6, -36], [mx1 + 15, 8, 0], [mx1 + 15, MAST_TOP + 6, -36]])
    cam.fit(fit_pts, margin=0.03)

    wall_quad = [(WALL_X[0], WALL_Y[1], 0), (WALL_X[1], WALL_Y[1], 0), (WALL_X[1], WALL_Y[0], 0), (WALL_X[0], WALL_Y[0], 0)]
    floor_quad = [(FLOOR_X[0], 0, FLOOR_Z[1]), (FLOOR_X[1], 0, FLOOR_Z[1]), (FLOOR_X[1], 0, FLOOR_Z[0]), (FLOOR_X[0], 0, FLOOR_Z[0])]
    floor = warp(floor_texture().convert("RGBA"), cam, floor_quad)
    bg = Image.new("RGBA", (W, H), (40, 34, 30, 255))
    bg.alpha_composite(floor)
    blank_img, full_img = bg.copy(), bg.copy()
    blank_img.alpha_composite(warp(Image.fromarray(wall).convert("RGBA"), cam, wall_quad))
    full_img.alpha_composite(warp(Image.fromarray(full).convert("RGBA"), cam, wall_quad))
    rail = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    draw_boxes(ImageDraw.Draw(rail), cam, rail_boxes(), SS)
    rail = rail.resize((W, H), Image.LANCZOS)
    blank_img.alpha_composite(rail)
    full_img.alpha_composite(rail)
    blank = np.asarray(blank_img.convert("RGB"), np.float32)
    full_px = np.asarray(full_img.convert("RGB"), np.float32)

    # Vignette for a camera-like falloff.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - 0.28 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    vig = vig[..., None]

    # Timeline: printer x-centre, head height and what has been printed.
    n_swaths = int(np.ceil((mx1 - mx0) / SWATH))
    frames = []
    start_x, first_x = mx0 - 110, mx0 + SWATH / 2
    for f in range(INTRO_FRAMES):
        frames.append((start_x + (first_x - start_x) * ease(f / (INTRO_FRAMES - 1)), my1, None))
    for s in range(n_swaths):
        sx0, sx1 = mx0 + s * SWATH, min(mx0 + (s + 1) * SWATH, mx1)
        cx = sx0 + SWATH / 2
        down = s % 2 == 0
        for f in range(PASS_FRAMES):
            p = (f + 1) / PASS_FRAMES
            y = my1 - (my1 - my0) * p if down else my0 + (my1 - my0) * p
            frames.append((cx, y, (sx0, sx1, down, y)))
        if s < n_swaths - 1:
            for f in range(STEP_FRAMES):
                frames.append((cx + SWATH * ease((f + 1) / STEP_FRAMES), y, None))
    last_x, last_y = frames[-1][0], frames[-1][1]
    park_x = mx1 + 260
    for f in range(EXIT_FRAMES):
        frames.append((last_x + (park_x - last_x) * ease(f / (EXIT_FRAMES - 1)), last_y, None))
    frames += [(None, None, None)] * HOLD_FRAMES

    wall_quad_mask = [(WALL_X[0], WALL_Y[1], 0), (WALL_X[1], WALL_Y[1], 0), (WALL_X[1], WALL_Y[0], 0), (WALL_X[0], WALL_Y[0], 0)]
    mask_tex = np.zeros(wall.shape[:2], np.uint8)

    def col(x):
        return int(np.searchsorted(xs, x))

    def row(y):
        return int(np.searchsorted(-ys, -y))

    stills = {int(i) for i in args.stills.split(",")} if args.stills else None
    ff = None if stills else subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", args.output],
        stdin=subprocess.PIPE,
    )
    mask_screen = np.zeros((H, W, 1), np.float32)
    for i, (cx, hy, printing) in enumerate(frames):
        if stills is not None and i > max(stills):
            break
        if printing:
            sx0, sx1, down, y = printing
            if down:
                mask_tex[r0:row(y), col(sx0):col(sx1)] = 255
            else:
                mask_tex[row(y):r1, col(sx0):col(sx1)] = 255
            m = warp(Image.fromarray(mask_tex), cam, wall_quad_mask, Image.BILINEAR)
            mask_screen = np.asarray(m, np.float32)[..., None] / 255
        if stills is not None and i not in stills:
            continue
        frame = blank * (1 - mask_screen) + full_px * mask_screen
        if cx is not None:
            boxes = printer_boxes(cx, hy)
            sh = np.asarray(shadow_layer(cam, boxes), np.float32)[..., None] / 255
            frame = frame * (1 - sh)
            layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
            draw_boxes(ImageDraw.Draw(layer), cam, boxes, SS)
            layer = layer.resize((W, H), Image.LANCZOS)
            a = np.asarray(layer, np.float32)
            frame = frame * (1 - a[..., 3:] / 255) + a[..., :3] * (a[..., 3:] / 255)
        frame = np.clip(frame * vig, 0, 255).astype(np.uint8)
        if stills is not None:
            Image.fromarray(frame).save(f"{args.output}.{i:04d}.jpg", quality=90)
            continue
        ff.stdin.write(frame.tobytes())
    if ff:
        ff.stdin.close()
        ff.wait()
    print(f"wrote {args.output}: {len(frames)} frames ({len(frames) / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
