#!/usr/bin/env python3
# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Render an asciinema v2 cast to an animated GIF without a browser or a Rust binary.

    python demo/render_cast.py demo.cast figure/demo_h200.gif [--font /path/mono.ttf]
        [--idle 2.0] [--fps 20] [--scale 1]

pyte emulates the terminal (SGR colours included), Pillow paints each frame in a monospace
face, identical consecutive frames are merged, and idle gaps are capped at --idle seconds so
a model load does not become a minute of nothing. Frame timing is the cast's own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pyte
from PIL import Image, ImageDraw, ImageFont

BG = (24, 25, 28)
FG = (222, 222, 216)
ANSI = {"black": (40, 42, 46), "red": (224, 108, 117), "green": (152, 195, 121),
        "brown": (229, 192, 123), "blue": (97, 175, 239), "magenta": (198, 120, 221),
        "cyan": (86, 182, 194), "white": (222, 222, 216), "default": FG}
BRIGHT = {"black": (92, 99, 112), "red": (240, 130, 140), "green": (170, 210, 140),
          "brown": (240, 205, 140), "blue": (120, 190, 250), "magenta": (215, 140, 240),
          "cyan": (110, 200, 210), "white": (255, 255, 255)}


def _color(name, bold=False, fallback=FG):
    if name == "default":
        return fallback
    if name in ANSI:
        return (BRIGHT if bold else ANSI).get(name, ANSI[name])
    if len(name) == 6:  # hex from 256/24-bit SGR
        try:
            return tuple(int(name[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return fallback


def _find_font(explicit):
    cands = [explicit] if explicit else []
    cands += ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise SystemExit("no monospace font found; pass --font")


def load_cast(path):
    with open(path) as f:
        head = json.loads(f.readline())
        events = [json.loads(l) for l in f if l.strip()]
    return head, [(float(t), kind, data) for t, kind, data in events if kind == "o"]


def render(cast, out, font_path=None, idle=2.0, fps=20, scale=1, px=14):
    head, events = load_cast(cast)
    cols, rows = int(head.get("width", 100)), int(head.get("height", 30))
    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)
    font = ImageFont.truetype(_find_font(font_path), px)
    cw = int(round(font.getlength("M"))); ch = int(px * 1.25)
    W, H = cols * cw + 16, rows * ch + 16

    def paint():
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for y in range(rows):
            line = screen.buffer[y]
            for x in range(cols):
                c = line[x]
                fg = _color(c.fg, c.bold); bg = _color(c.bg, False, BG)
                if c.reverse:
                    fg, bg = bg, fg
                if bg != BG:
                    d.rectangle([8 + x * cw, 8 + y * ch, 8 + (x + 1) * cw, 8 + (y + 1) * ch], fill=bg)
                if c.data and c.data != " ":
                    d.text((8 + x * cw, 8 + y * ch), c.data, font=font, fill=fg)
        if scale != 1:
            img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        return img

    frames, durations = [], []
    t_prev = 0.0; last_img = None; step = 1.0 / fps
    # coalesce output events closer than one frame period (but never past one period of
    # accumulated output, or a dense token stream would collapse into one frame); cap idle gaps
    pending = 0.0
    for i, (t, _, data) in enumerate(events):
        gap = min(max(t - t_prev, 0.0), idle); t_prev = t
        pending += gap
        stream.feed(data.encode("utf-8", "replace"))
        nxt = events[i + 1][0] if i + 1 < len(events) else None
        if nxt is not None and (nxt - t) < step and pending < step:
            continue                       # more output arrives within this frame period
        img = paint()
        if last_img is not None and img.tobytes() == last_img.tobytes():
            durations[-1] += pending; pending = 0.0
            continue
        frames.append(img); durations.append(max(pending, step)); pending = 0.0
        last_img = img
    durations[-1] += 3.0                   # hold the final screen
    pal = [f.quantize(colors=64, method=Image.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], optimize=True, loop=0,
                duration=[int(d * 1000) for d in durations], disposal=2)
    size = os.path.getsize(out) / 1e6
    print(f"RENDER_CAST_DONE {out} frames={len(frames)} {W}x{H} {size:.2f} MB "
          f"span={sum(durations):.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cast"); ap.add_argument("out")
    ap.add_argument("--font"); ap.add_argument("--idle", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=20); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--px", type=int, default=14)
    a = ap.parse_args()
    render(a.cast, a.out, a.font, a.idle, a.fps, a.scale, a.px)


if __name__ == "__main__":
    main()
