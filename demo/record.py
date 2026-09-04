#!/usr/bin/env python3
# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Record a command's terminal output as an asciinema v2 cast, no asciinema needed.

    python demo/record.py --cols 100 --rows 40 -o demo.cast -- bash demo/demo_session.sh

The command runs in a pseudo-terminal of the given size; every chunk it writes is stored with
the wall-clock offset at which it arrived. `demo/render_cast.py` turns the cast into a GIF.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import select
import struct
import sys
import termios
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--rows", type=int, default=30)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("no command given")
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"], os.environ["LINES"] = str(a.cols), str(a.rows)
        os.execvp(cmd[0], cmd)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", a.rows, a.cols, 0, 0))
    t0 = time.time()
    with open(a.out, "w") as f:
        f.write(json.dumps({"version": 2, "width": a.cols, "height": a.rows,
                            "timestamp": int(t0), "env": {"TERM": "xterm-256color"}}) + "\n")
        while True:
            r, _, _ = select.select([fd], [], [], 1.0)
            if fd not in r:
                continue
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            text = data.decode("utf-8", "replace")
            f.write(json.dumps([round(time.time() - t0, 4), "o", text]) + "\n")
            f.flush()
            sys.stdout.write(text)
            sys.stdout.flush()
    _, status = os.waitpid(pid, 0)
    rc = os.waitstatus_to_exitcode(status)
    print(f"\nRECORD_DONE {a.out} rc={rc}", flush=True)
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
