#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render an asciinema v2 .cast (a genuine terminal recording) to an animated GIF by replaying
its output stream through a pyte terminal emulator and rasterizing each frame with PIL. No
ffmpeg/agg needed. The .cast is the real capture; this just draws what the terminal showed.

Usage: python figure/demo_cast_to_gif.py --cast demo.cast --out assets/demo_race.gif --fps 12
"""
import argparse
import json
import os

import pyte
from PIL import Image, ImageDraw, ImageFont

BG = (13, 17, 23)
FG_DEFAULT = (215, 221, 227)
COLORS = {  # pyte color name -> RGB
    "default": FG_DEFAULT, "white": (236, 240, 244), "black": (30, 34, 40),
    "green": (55, 209, 138), "cyan": (122, 184, 230), "magenta": (239, 120, 182),
    "blue": (74, 163, 199), "yellow": (229, 192, 123), "red": (224, 108, 117),
    "brown": (229, 192, 123), "brightblack": (120, 130, 140),
}
FONTS = ["/System/Library/Fonts/Menlo.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/Library/Fonts/DejaVuSansMono.ttf",
         "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"]


def _font(sz):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def _rgb(c):
    if isinstance(c, str) and len(c) == 6:
        try:
            return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return COLORS.get(c, FG_DEFAULT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cast", required=True)
    ap.add_argument("--out", default="assets/demo_race.gif")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--fontsize", type=int, default=15)
    ap.add_argument("--cols", type=int, default=0, help="override grid width (0=cast header)")
    ap.add_argument("--rows", type=int, default=0, help="override grid height (0=cast header)")
    ap.add_argument("--start-substr", default="", help="begin rendering when this text first appears")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="cap rendered duration (0=all)")
    args = ap.parse_args()

    lines = open(args.cast).read().splitlines()
    header = json.loads(lines[0])
    cols = args.cols or int(header.get("width", 118))
    rows = args.rows or int(header.get("height", 34))
    events = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        t, kind, data = json.loads(ln)
        if kind == "o":
            events.append((float(t), data))
    if not events:
        raise SystemExit("no output events in cast")
    t0 = 0.0
    if args.start_substr:
        acc = ""
        for t, data in events:
            acc += data
            if args.start_substr in acc:
                t0 = t
                break
    events = [(t - t0, d) for t, d in events if t >= t0]
    duration = events[-1][0]
    if args.max_seconds:
        duration = min(duration, args.max_seconds)

    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)
    font = _font(args.fontsize)
    bb = font.getbbox("M"); cw = bb[2] - bb[0] or args.fontsize * 0.6
    ch = int(args.fontsize * 1.32)
    cw = int(cw)
    W, H = cols * cw, rows * ch

    frames = []
    n = max(1, int(duration * args.fps))
    ei = 0
    for k in range(n + 1):
        tk = (k / args.fps)
        while ei < len(events) and events[ei][0] <= tk:
            stream.feed(events[ei][1])
            ei += 1
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img)
        buf = screen.buffer
        for y in range(rows):
            row = buf.get(y, {})
            for x in range(cols):
                cell = row.get(x)
                if cell is None or cell.data in (" ", ""):
                    continue
                col = _rgb(cell.fg if cell.fg != "default" else "default")
                if getattr(cell, "bold", False) and cell.fg == "default":
                    col = (240, 244, 248)
                dr.text((x * cw, y * ch), cell.data, font=font, fill=col)
        frames.append(img)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], loop=0,
                   duration=int(1000 / args.fps), optimize=True)
    print(f"DEMO_GIF {args.out} {os.path.getsize(args.out)/1e6:.2f}MB frames={len(frames)} {cols}x{rows}")


if __name__ == "__main__":
    main()
