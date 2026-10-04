"""Composite a photoreal video of a rail-mounted vertical wall printer.

Locked-off camera, like a real job filmed on a tripod. The printer stands on a
fixed floor rail in front of a framed wall panel. Its print head runs up and
down the mast, printing one vertical strip per pass; the image appears only in
the strip the head has covered, and after each pass the printer steps one strip
to the right. When the print is done the printer rolls along the rail out of
the way and the shot holds on the finished piece.

Inputs (see static/uploads/wall_printer/rail/):
    plate_1080.jpg  empty room, 1920x1080, blank panel and empty rail
    printer.png     printer body (no head), keyed RGBA
    head.png        print head unit, keyed RGBA
    mural_4x3.jpg   the artwork to print

Usage:
    python scripts/wall_printer_rail.py static/uploads/wall_printer/rail OUTPUT.mp4
    python scripts/wall_printer_rail.py static/uploads/wall_printer/home OUTPUT.mp4 --scene home

Requires Pillow, numpy and ffmpeg on PATH.
"""
import argparse
import os
import subprocess

import numpy as np
from PIL import Image, ImageFilter

W, H = 1920, 1080
FPS = 30
SPRITE_MAST_X = 122  # left edge of the mast in printer.png
HEAD_BOX_FRAC = 0.78  # print box part of head.png (the rest is the bracket)

# Per-scene layout, measured on each plate_1080.jpg.
SCENES = {
    # Hotel lobby: print into a framed white panel.
    "lobby": dict(panel=(723, 206, 1468, 781), floor_y=995, printer_h=790, swath=48,
                  park=1.4, wall_bottom=960, plain_wall=False),
    # Home: print straight onto a plain painted wall, no frame.
    "home": dict(panel=(685, 290, 1339, 781), floor_y=949, printer_h=694, swath=42,
                 park=1.2, wall_bottom=880, plain_wall=True),
}
PANEL, FLOOR_Y, PRINTER_H, SWATH, PARK, WALL_BOTTOM, PLAIN_WALL = (723, 206, 1468, 781), 995, 790, 48, 1.4, 960, False
PASS_FRAMES = 14
STEP_FRAMES = 5
START_HOLD = 20
EXIT_FRAMES = 40
END_HOLD = 75


def ease(t):
    return t * t * (3 - 2 * t)


def load_assets(d, sprites):
    plate = np.asarray(Image.open(os.path.join(d, "plate_1080.jpg")).convert("RGB"), np.float32)
    printer = Image.open(os.path.join(sprites, "printer.png")).convert("RGBA")
    s = PRINTER_H / printer.height
    printer = printer.resize((round(printer.width * s), PRINTER_H), Image.LANCZOS)
    mast_x = SPRITE_MAST_X * s
    head = Image.open(os.path.join(sprites, "head.png")).convert("RGBA")
    hs = SWATH / (head.width * HEAD_BOX_FRAC)
    head = head.resize((round(head.width * hs), round(head.height * hs)), Image.LANCZOS)

    x0, y0, x1, y1 = PANEL
    ink = Image.open(os.path.join(d if os.path.exists(os.path.join(d, "mural_4x3.jpg")) else sprites, "mural_4x3.jpg")).convert("RGB")
    # Cover-fit the artwork to the panel.
    pw, ph = x1 - x0, y1 - y0
    k = max(pw / ink.width, ph / ink.height)
    ink = ink.resize((round(ink.width * k), round(ink.height * k)), Image.LANCZOS)
    cx, cy = (ink.width - pw) // 2, (ink.height - ph) // 2
    ink = np.asarray(ink.crop((cx, cy, cx + pw, cy + ph)), np.float32)
    panel = plate[y0:y1, x0:x1]
    printed = plate.copy()
    if PLAIN_WALL:
        # Printed straight onto the wall: drop the artwork's own plaster margin so
        # only the window is inked, and tint it with the wall's colour and light.
        edge = np.concatenate([ink[:6].reshape(-1, 3), ink[-6:].reshape(-1, 3), ink[:, :6].reshape(-1, 3), ink[:, -6:].reshape(-1, 3)])
        bg = np.median(edge, 0)
        alpha = np.clip((np.abs(ink - bg).sum(-1) - 45) / 40, 0, 1)
        # Keep only the window itself: ignore lines along the artwork's outer edge
        # and fade out its margin.
        by, bx = int(alpha.shape[0] * 0.015), int(alpha.shape[1] * 0.012)
        alpha[:by], alpha[-by:], alpha[:, :bx], alpha[:, -bx:] = 0, 0, 0, 0
        rows = np.where((alpha > 0.5).mean(1) > 0.3)[0]
        cols = np.where((alpha > 0.5).mean(0) > 0.3)[0]
        keep = np.zeros(alpha.shape, np.float32)
        keep[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = 1
        keep = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)), np.float32) / 255
        # Inside the window's bounds use a softer key so pale parts (the header
        # beam, sill highlights) and the painted shadows print in full.
        soft = np.clip((np.abs(ink - bg).sum(-1) - 18) / 30, 0, 1)
        soft[:by], soft[-by:], soft[:, :bx], soft[:, -bx:] = 0, 0, 0, 0
        alpha = np.maximum(alpha, soft) * keep
        alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)), np.float32)[..., None] / 255
        inked = ink * panel / np.maximum(bg, 1)
        printed[y0:y1, x0:x1] = panel * (1 - alpha) + inked * alpha
    else:
        # Ink sits on the panel: multiply by the panel's own lighting.
        printed[y0:y1, x0:x1] = ink * np.clip(panel / 242.0, 0, 1.05)
    return plate, printed, printer, mast_x, head


def timeline():
    """Yield (strip_index, head_y_fraction, printer_x_strip, revealed) per frame."""
    x0, y0, x1, y1 = PANEL
    n = int(np.ceil((x1 - x0) / SWATH))
    frames = []
    for _ in range(START_HOLD):
        frames.append(dict(pos=0.0, hy=0.0, strip=None))
    hy = 0.0
    for s in range(n):
        down = s % 2 == 0
        for f in range(PASS_FRAMES):
            p = (f + 1) / PASS_FRAMES
            hy = p if down else 1 - p
            frames.append(dict(pos=float(s), hy=hy, strip=(s, down, hy)))
        if s < n - 1:
            for f in range(STEP_FRAMES):
                frames.append(dict(pos=s + ease((f + 1) / STEP_FRAMES), hy=hy, strip=None))
    last = float(n - 1)
    park = last + PARK  # right-hand end of the floor rail
    for f in range(EXIT_FRAMES):
        frames.append(dict(pos=last + (park - last) * ease((f + 1) / EXIT_FRAMES), hy=hy, strip=None))
    for _ in range(END_HOLD):
        frames.append(dict(pos=park, hy=hy, strip=None))
    return frames, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assets", help="scene folder with plate_1080.jpg (and optionally mural_4x3.jpg)")
    ap.add_argument("output")
    ap.add_argument("--scene", choices=sorted(SCENES), default="lobby")
    ap.add_argument("--sprites", default="static/uploads/wall_printer/rail", help="folder with printer.png, head.png")
    ap.add_argument("--stills", help="comma-separated frame numbers to save as JPEGs")
    args = ap.parse_args()
    global PANEL, FLOOR_Y, PRINTER_H, SWATH, PARK, WALL_BOTTOM, PLAIN_WALL
    sc = SCENES[args.scene]
    PANEL, FLOOR_Y, PRINTER_H, SWATH = sc["panel"], sc["floor_y"], sc["printer_h"], sc["swath"]
    PARK, WALL_BOTTOM, PLAIN_WALL = sc["park"], sc["wall_bottom"], sc["plain_wall"]

    plate, printed, printer, mast_x, head = load_assets(args.assets, args.sprites)
    x0, y0, x1, y1 = PANEL
    frames, n = timeline()
    stills = {int(i) for i in args.stills.split(",")} if args.stills else None

    pa = np.asarray(printer, np.float32)
    shadow_src = Image.fromarray(pa[..., 3].astype(np.uint8)).filter(ImageFilter.GaussianBlur(10))
    head_h = head.height
    box_w = SWATH
    gap = head.width - box_w  # bracket between print box and mast
    y_top, y_bot = y0 + head_h * 0.5 - 6, y1 - head_h * 0.5 + 6  # head centre travel
    rng = np.random.default_rng(5)

    mask = np.zeros((H, W), np.float32)
    ff = None
    if stills is None:
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", args.output],
            stdin=subprocess.PIPE,
        )
    for i, fr in enumerate(frames):
        hy_px = y_top + (y_bot - y_top) * fr["hy"]
        if fr["strip"]:
            s, down, _ = fr["strip"]
            sx0, sx1 = x0 + s * SWATH, min(x0 + (s + 1) * SWATH, x1)
            # Paint what the nozzles have passed; the end of a pass covers the whole strip.
            done = fr["hy"] in (0.0, 1.0)
            if down:
                edge = y1 if done else int(round(hy_px + head_h * 0.5 - 4))
                mask[y0:min(edge, y1), sx0:sx1] = 1
            else:
                edge = y0 if done else int(round(hy_px - head_h * 0.5 + 4))
                mask[max(edge, y0):y1, sx0:sx1] = 1
        if stills is not None and i not in stills:
            continue

        frame = plate * (1 - mask[..., None]) + printed * mask[..., None]
        strip_x = x0 + fr["pos"] * SWATH  # left edge of the print box
        mast_left = strip_x + box_w + gap - 4
        px = int(round(mast_left - mast_x))
        py = FLOOR_Y - printer.height
        img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).convert("RGBA")

        # Soft shadow on the wall, cast down-right by the picture lights above.
        sh = Image.new("L", (W, H), 0)
        sh.paste(shadow_src, (px + 16, py + 22))
        sh_a = np.asarray(sh, np.float32)[..., None] / 255 * 0.28
        sh_a[WALL_BOTTOM:] = 0
        base = np.asarray(img, np.float32)
        base[..., :3] *= 1 - sh_a
        # Contact shadow under the printer on the floor.
        yy, xx = np.ogrid[0:H, 0:W]
        cxs = px + printer.width * 0.42
        contact = np.exp(-(((xx - cxs) / (printer.width * 0.55)) ** 2 + ((yy - FLOOR_Y + 4) / 10) ** 2))
        base[..., :3] *= 1 - 0.45 * contact[..., None]

        printing = fr["strip"] is not None
        if printing:
            # Violet UV glow on the wall at the nozzle line, with a slight flicker.
            gx, gy = strip_x + box_w / 2, hy_px + head_h * 0.45
            g = np.exp(-(((xx - gx) / 38) ** 2 + ((yy - gy) / 16) ** 2)) * (0.75 + 0.25 * rng.random())
            base[..., :3] += g[..., None] * np.array([95, 45, 190], np.float32)
        img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        img = img.convert("RGBA")
        img.alpha_composite(printer, (px, py))
        img.alpha_composite(head, (int(round(strip_x)), int(round(hy_px - head_h / 2))))
        out = np.asarray(img.convert("RGB"))
        if stills is not None:
            Image.fromarray(out).save(f"{args.output}.{i:04d}.jpg", quality=92)
            continue
        ff.stdin.write(out.tobytes())
    if ff:
        ff.stdin.close()
        ff.wait()
    print(f"{len(frames)} frames ({len(frames) / FPS:.1f}s), {n} strips")


if __name__ == "__main__":
    main()
