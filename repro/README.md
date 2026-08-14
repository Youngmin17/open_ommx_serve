# Repro provenance — per-GPU orchestration scripts

The concrete per-GPU scripts used to produce `figure/RESULTS.md` + the measured figures.
Each runs every method in its own env on ONE assigned GPU (`CUDA_VISIBLE_DEVICES`).

- `orchestrate_h200.sh` — NVIDIA H200 NVL (fig1). Runs bf16 / OMMX / KIVI / Kitty, each in its own conda env or venv. Override `REPO`, `CONDASH`, the `ENV_*`
  conda env names, `KITTY_SRC`, `KVENV`, `HF_HOME` for your layout (see the script header).
- `orchestrate_a100.sh` — NVIDIA A100-SXM4-80GB (fig2). Installable subset (bf16 / OMMX /
  KIVI) over per-method venvs. Override `REPO`, `OMMXPY`, `KIVIPY`.

Both assume model / Kitty artifacts are already present locally (`HF_HUB_OFFLINE=1`).
The portable single-venv entry point for a fresh host is `../run.sh`; these scripts are the
concrete multi-env instances behind the two published figures.
