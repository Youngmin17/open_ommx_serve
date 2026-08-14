#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live 4-pane terminal TUI (rich) that streams the FOUR real per-token generation logs
(figure/data_demo/<m>.json from demo_gen.py) side by side at their real relative timing,
time-compressed to a watchable clip. Recorded verbatim by asciinema (a genuine terminal
capture), then rendered to GIF by demo_cast_to_gif.py.

Usage (under asciinema):
  asciinema rec -c "python figure/demo_tui.py --src figure/data_demo --seconds 22" demo.cast
"""
import argparse
import json
import os
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PANES = [("ommx", "OMMX (HF-eager)", "green"),
         ("bf16", "bf16 (HF-eager)", "cyan"),
         ("kitty", "Kitty (HF-eager)", "magenta"),
         ("kivi", "KIVI (HF-eager)", "blue")]
COLS, ROWS = 118, 34          # forced console size (pyte renders at the same grid)


def load(src):
    out = {}
    for k, _, _ in PANES:
        p = os.path.join(src, f"{k}.json")
        if os.path.exists(p):
            out[k] = json.load(open(p))
    return out


def text_upto(evts, t):
    s = []
    for piece, ct in evts:
        if ct > t:
            break
        s.append(piece)
    return "".join(s), len(s)


def pane(label, color, d, treal, body_h, body_w):
    if d is None:
        return Panel(Text("n/a"), title=label, border_style="dim")
    txt, n = text_upto(d["events"], treal)
    done = treal >= d["decode_s"] or n >= d["n_out"]
    if done:
        n, rate = d["n_out"], d["tok_per_s"]
    else:
        rate = n / treal if treal > 0 else 0.0
    # keep only the last body_h wrapped lines so the panel never overflows
    import textwrap
    wrapped = []
    for para in txt.split("\n"):
        wrapped += textwrap.wrap(para, body_w) or [""]
    body = Text("\n".join(wrapped[-body_h:]), style="grey85")
    tag = "[bold green]✓ done[/]" if done else "[yellow]▮ streaming[/]"
    foot = f"[bold]{n:4d}[/] tok · [bold]{rate:5.1f}[/] tok/s · {tag}"
    return Panel(body, title=f"[bold {color}]{label}[/]", subtitle=foot,
                 subtitle_align="left", border_style=color, height=body_h + 2)


def render(data, treal, speed):
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1); grid.add_column(ratio=1)
    bh, bw = (ROWS - 1) // 2 - 4, COLS // 2 - 4   # 2*(bh+4)+1 header <= ROWS (footer stays visible)
    cells = [pane(lbl, col, data.get(k), treal, bh, bw) for k, lbl, col in PANES]
    grid.add_row(cells[0], cells[1]); grid.add_row(cells[2], cells[3])
    head = Text.from_markup(
        f"[bold]B=1 decode race — Llama-3.1-8B, 32K context[/]   "
        f"t={treal:5.1f}s real · ×{speed:.0f} speed", justify="center")
    return Group(head, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="figure/data_demo")
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    data = load(args.src)
    if not data:
        raise SystemExit(f"no demo logs in {args.src}")
    real_max = max(d["decode_s"] for d in data.values())
    speed = real_max / args.seconds
    console = Console(width=COLS, height=ROWS, force_terminal=True)
    frames = int(args.seconds * args.fps)
    dt = 1.0 / args.fps
    with Live(render(data, 0.0, speed), console=console, refresh_per_second=args.fps,
              screen=False) as live:
        for i in range(frames + 1):
            live.update(render(data, (i * dt) * speed, speed))
            time.sleep(dt)


if __name__ == "__main__":
    main()
