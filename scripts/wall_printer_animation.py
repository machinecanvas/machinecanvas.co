"""Render an MP4 of a vertical wall printer printing an image onto a wall.

The printer stands on the floor in front of the wall. Its print head runs up and
down a vertical mast, laying down one vertical swath of the image per pass, then
the whole machine steps sideways to the next swath. Once the mural is finished
the printer rolls out of frame and the camera holds on the result.

Usage:
    python scripts/wall_printer_animation.py IMAGE OUTPUT.mp4 [--fps 30]

Requires Pillow, numpy and ffmpeg on PATH.
"""
import argparse
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
FLOOR_Y = 975
MURAL_BOX = (190, 70, 1730, 930)  # left, top, right, bottom on the wall
SWATH = 70  # px width printed per vertical pass
PASS_FRAMES = 18  # frames for the head to travel the mural height
STEP_FRAMES = 5  # frames to shift sideways to the next swath
INTRO_FRAMES = 45
EXIT_FRAMES = 50
HOLD_FRAMES = 90

MAST_TOP = 30
HEAD_H = 46


def make_wall(wall_rgb):
    rng = np.random.default_rng(7)
    wall = np.empty((H, W, 3), np.float32)
    wall[:] = wall_rgb
    # Subtle plaster texture plus a soft light falloff from the top-left.
    noise = rng.normal(0, 3.5, (H // 4, W // 4)).astype(np.float32)
    noise = np.array(Image.fromarray(noise).resize((W, H), Image.BICUBIC))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1.06 - 0.12 * (xx / W) - 0.08 * (yy / H)
    wall = wall * light[..., None] + noise[..., None]

    # Skirting board and wooden floor.
    wall[FLOOR_Y - 28:FLOOR_Y] = (236, 232, 224)
    wall[FLOOR_Y - 2:FLOOR_Y] = (190, 184, 174)
    floor = np.zeros((H - FLOOR_Y, W, 3), np.float32)
    floor[:] = (150, 108, 72)
    for i, x in enumerate(range(-80, W, 240)):
        floor[:, max(x, 0):max(x, 0) + 2] = (110, 76, 48)
        floor[:, max(x, 0) + 2:x + 240] *= 0.94 + 0.06 * ((i * 37) % 5) / 4
    floor *= np.linspace(1.0, 0.8, H - FLOOR_Y)[:, None, None]
    wall[FLOOR_Y:] = floor
    return np.clip(wall, 0, 255).astype(np.uint8)


def load_mural(path, wall):
    l, t, r, b = MURAL_BOX
    img = Image.open(path).convert("RGB").resize((r - l, b - t), Image.LANCZOS)
    mural = wall.copy()
    ink = np.asarray(img, np.float32)
    # Pixels close to the image's own background colour are left as bare wall,
    # so the mural blends into the plaster instead of showing a rectangle.
    bg = np.concatenate([ink[:8].reshape(-1, 3), ink[-8:].reshape(-1, 3)]).mean(0)
    dist = np.abs(ink - bg).sum(-1)
    alpha = np.clip((dist - 35) / 40, 0, 1)
    alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    under = wall[t:b, l:r].astype(np.float32)
    # Re-tint the image so its plaster matches the wall underneath; this also
    # carries the wall's texture and lighting into the printed ink.
    printed = ink * under / bg
    mural[t:b, l:r] = np.clip(under * (1 - alpha[..., None]) + printed * alpha[..., None], 0, 255).astype(np.uint8)
    return mural


def make_printer():
    """Return (RGBA sprite, x offset of swath left edge in sprite, shadow sprite)."""
    sw, sh = 260, H
    spr = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    d = ImageDraw.Draw(spr)
    cx = sw // 2
    # Mast: brushed aluminium extrusion.
    for i in range(34):
        shade = int(150 + 70 * np.sin(np.pi * i / 33))
        d.line([(cx - 17 + i, MAST_TOP), (cx - 17 + i, FLOOR_Y - 40)], fill=(shade, shade, shade + 6, 255))
    d.rectangle([cx - 17, MAST_TOP, cx + 16, MAST_TOP + 14], fill=(55, 58, 64, 255))
    for y in range(MAST_TOP + 30, FLOOR_Y - 60, 60):
        d.line([(cx - 5, y), (cx + 5, y)], fill=(120, 124, 130, 255), width=2)
    # Base with wheels.
    d.rounded_rectangle([cx - 110, FLOOR_Y - 70, cx + 110, FLOOR_Y - 18], 10, fill=(40, 44, 52, 255))
    d.rounded_rectangle([cx - 104, FLOOR_Y - 64, cx + 104, FLOOR_Y - 58], 3, fill=(66, 165, 245, 255))
    d.text((cx - 70, FLOOR_Y - 50), "MACHINE CANVAS", fill=(220, 225, 232, 255))
    for wx in (cx - 85, cx + 85):
        d.ellipse([wx - 17, FLOOR_Y - 34, wx + 17, FLOOR_Y], fill=(25, 25, 28, 255))
        d.ellipse([wx - 6, FLOOR_Y - 23, wx + 6, FLOOR_Y - 11], fill=(150, 150, 155, 255))
    shadow = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle([cx - 12, MAST_TOP + 10, cx + 28, FLOOR_Y - 30], fill=(0, 0, 0, 70))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    return spr, shadow


def make_head():
    hw = SWATH + 56
    head = Image.new("RGBA", (hw, HEAD_H + 30), (0, 0, 0, 0))
    d = ImageDraw.Draw(head)
    d.rounded_rectangle([0, 0, hw - 1, HEAD_H], 8, fill=(32, 35, 42, 255))
    d.rounded_rectangle([4, 4, hw - 5, 14], 4, fill=(58, 62, 72, 255))
    for i, c in enumerate([(0, 174, 239), (236, 0, 140), (255, 230, 0), (20, 20, 20), (240, 240, 240)]):
        d.rectangle([12 + i * 16, 20, 22 + i * 16, 34], fill=c + (255,))
    d.ellipse([hw - 22, 20, hw - 12, 30], fill=(80, 255, 140, 255))
    return head


def ease(t):
    return t * t * (3 - 2 * t)


def main():
    global SWATH, PASS_FRAMES, STEP_FRAMES, HOLD_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("output")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--swath", type=int, default=SWATH, help="px printed per vertical pass")
    ap.add_argument("--pass-frames", type=int, default=PASS_FRAMES)
    ap.add_argument("--step-frames", type=int, default=STEP_FRAMES)
    ap.add_argument("--hold-frames", type=int, default=HOLD_FRAMES)
    args = ap.parse_args()
    SWATH, PASS_FRAMES, STEP_FRAMES, HOLD_FRAMES = args.swath, args.pass_frames, args.step_frames, args.hold_frames

    src = Image.open(args.image).convert("RGB")
    corner = np.asarray(src.crop((5, 5, 60, 60)), np.float32).reshape(-1, 3).mean(0)
    wall = make_wall(corner)
    mural = load_mural(args.image, wall)
    printer, shadow = make_printer()
    head = make_head()  # sized from SWATH, so built after args are applied
    l, t, r, b = MURAL_BOX
    n_swaths = int(np.ceil((r - l) / SWATH))
    mast_off = printer.width // 2  # mast centre in sprite coordinates

    # Build a timeline of (printer x-centre, head y, printed mask updates).
    mask = np.zeros((H, W), bool)
    frames = []  # (mast_x, head_y, printing, swath_rect, progress_y, direction)
    start_x = -150
    first_x = l + SWATH / 2
    for f in range(INTRO_FRAMES):
        frames.append((start_x + (first_x - start_x) * ease(f / (INTRO_FRAMES - 1)), t, None))
    for s in range(n_swaths):
        sx0 = l + s * SWATH
        sx1 = min(sx0 + SWATH, r)
        mx = sx0 + SWATH / 2
        down = s % 2 == 0
        for f in range(PASS_FRAMES):
            p = (f + 1) / PASS_FRAMES
            y = t + (b - t) * p if down else b - (b - t) * p
            frames.append((mx, y, (sx0, sx1, down, y)))
        if s < n_swaths - 1:
            for f in range(STEP_FRAMES):
                frames.append((mx + SWATH * ease((f + 1) / STEP_FRAMES), y, None))
    last_x, last_y = frames[-1][0], frames[-1][1]
    for f in range(EXIT_FRAMES):
        frames.append((last_x + (W + 200 - last_x) * ease(f / (EXIT_FRAMES - 1)), last_y, None))
    for f in range(HOLD_FRAMES):
        frames.append((None, None, None))

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", args.output],
        stdin=subprocess.PIPE,
    )
    hw = head.width
    for mx, hy, printing in frames:
        if printing:
            sx0, sx1, down, y = printing
            yi = int(round(y))
            if down:
                mask[t:yi, sx0:sx1] = True
            else:
                mask[yi:b, sx0:sx1] = True
        frame = np.where(mask[..., None], mural, wall)
        img = Image.fromarray(frame).convert("RGBA")
        if mx is not None:
            px = int(round(mx)) - mast_off
            img.alpha_composite(shadow, (px + 18, 0))
            img.alpha_composite(printer, (px, 0))
            hy_i = int(round(hy)) - HEAD_H // 2
            if printing:
                # Faint ink-mist glow along the edge being printed.
                glow = Image.new("RGBA", (hw, 18), (120, 200, 255, 0))
                gd = ImageDraw.Draw(glow)
                gy = HEAD_H // 2 + 2 if printing[2] else -HEAD_H // 2 - 14
                for i in range(9):
                    gd.rectangle([6, i, hw - 7, 17 - i], fill=(150, 210, 255, 10))
                img.alpha_composite(glow, (int(mx) - hw // 2, int(hy) + gy))
            img.alpha_composite(head, (int(round(mx)) - hw // 2, hy_i))
        ff.stdin.write(img.convert("RGB").tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.output}: {len(frames)} frames ({len(frames) / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
