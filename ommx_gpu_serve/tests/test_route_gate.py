# Copyright (c) 2024-2026, OMMX Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pin the route-evidence gate against the one failure it used to let through.

The gate exists so a CUSTOM arm that did not actually run the OMMX kernel cannot surface as
an OMMX bar. It classifies failure tags by SUFFIX (``*_NOFIRE`` / ``*_DEAD``) precisely so a
newly added failure tag cannot be silently ignored — but ``DECODE_BF16_UNROUTED`` matches
neither suffix, so it landed in the informational ``other_tags`` bucket and the arm stayed
``ok=True``.

That is not a hypothetical. Measured on H200 with the published bar's own environment
(``OMMX_ATTN_GRAPH=1``), a ``--batches 1,4`` run of ``abl_attn`` produced:

    GRAPH_ROUTE_FIRED    rank=0 pid=... fmt=i2f4 ctx=129
    DECODE_BF16_UNROUTED rank=0 pid=... B=4 graph=True: a uniform single-token decode of
      4 requests was served by bf16 FlashAttention (no OMMX route accepts it ...)

and the b4 cells came back ``ok=True, route_fired=True`` carrying bf16 timings — the exact
"bf16 published as OMMX" outcome the gate is for. The demotion has to be PER BATCH SIZE, not
per arm: the same worker process routed B=1 correctly, and discarding those cells too would
throw away a valid measurement to punish an invalid one.
"""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The verbatim sentinel from the H200 probe run (runs/e2e_h200batch_arms/abl_attn.fire.log).
REAL_SENTINEL = (
    "GRAPH_ROUTE_FIRED rank=0 pid=2989824 fmt=i2f4 ctx=129\n"
    "DECODE_BF16_UNROUTED rank=0 pid=2989824 B=4 graph=True: a uniform single-token decode "
    "of 4 requests was served by bf16 FlashAttention (no OMMX route accepts it: the batched "
    "pool is not active for this step and the single-batch route would answer only request "
    "0). This run's OMMX coverage is PARTIAL - see ommx_route_health()"
    "['unrouted_decode_b'].\n"
)


def _bench():
    path = os.path.join(REPO, "ommx_gpu_serve", "bench", "bench_e2e_a100.py")
    spec = importlib.util.spec_from_file_location("bench_e2e_a100", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _evidence(tmp_path, text):
    f = tmp_path / "abl_attn.fire.log"
    f.write_text(text)
    return _bench()._read_fire_evidence(str(f))


def test_unrouted_batch_is_extracted_from_the_real_sentinel(tmp_path):
    ev = _evidence(tmp_path, REAL_SENTINEL)
    assert ev["unrouted_batches"] == [4]
    # The arm as a whole is still a valid OMMX measurement -- B=1 routed and proved it.
    assert ev["ok"] is True
    assert "DECODE_BF16_UNROUTED" in ev["other_tags"]


def test_a_clean_arm_reports_no_unrouted_batches(tmp_path):
    ev = _evidence(tmp_path, "GRAPH_ROUTE_FIRED rank=0 pid=1 fmt=i2f4 ctx=129\n")
    assert ev["unrouted_batches"] == []
    assert ev["ok"] is True


def test_multiple_unrouted_batch_sizes_are_all_captured(tmp_path):
    ev = _evidence(tmp_path, (
        "GRAPH_ROUTE_FIRED rank=0 pid=1 fmt=i2f4 ctx=129\n"
        "DECODE_BF16_UNROUTED rank=0 pid=1 B=2 graph=True: served by bf16\n"
        "DECODE_BF16_UNROUTED rank=0 pid=1 B=8 graph=True: served by bf16\n"))
    assert ev["unrouted_batches"] == [2, 8]


def test_a_malformed_batch_token_does_not_crash_the_parser(tmp_path):
    """Provenance parsing must never be able to take a measurement down."""
    ev = _evidence(tmp_path, (
        "GRAPH_ROUTE_FIRED rank=0 pid=1 fmt=i2f4 ctx=129\n"
        "DECODE_BF16_UNROUTED rank=0 pid=1 B=notanumber graph=True: served by bf16\n"))
    assert ev["unrouted_batches"] == []
    assert ev["ok"] is True


def test_nofire_suffix_rule_still_demotes_the_whole_arm(tmp_path):
    """The per-batch path must not weaken the existing arm-level gate."""
    ev = _evidence(tmp_path, "GRAPH_ROUTE_DEAD rank=0 pid=1 reason=x\n")
    assert ev["ok"] is False
    assert "GRAPH_ROUTE_DEAD" in ev["nofire"]


def test_demotion_hits_only_the_unrouted_batch_sizes():
    """The orchestrator's per-batch demotion, applied to the real cell set.

    Mirrors the loop in bench_e2e_a100's gating block: cells are keyed ``b<batch>_ctx<n>``,
    and only the batch sizes the backend named may be demoted.
    """
    unrouted = [4]
    cells = {"b1_ctx1024": {"ok": True, "route_fired": True},
             "b1_ctx4096": {"ok": True, "route_fired": True},
             "b4_ctx1024": {"ok": True, "route_fired": True},
             "b4_ctx4096": {"ok": True, "route_fired": True}}
    for b in unrouted:
        for name, cell in cells.items():
            if name.startswith(f"b{b}_"):
                cell["ok"] = False
                cell["route_fired"] = False
    assert [n for n, c in sorted(cells.items()) if c["ok"]] == ["b1_ctx1024", "b1_ctx4096"]
    assert [n for n, c in sorted(cells.items()) if not c["ok"]] == ["b4_ctx1024", "b4_ctx4096"]
    # A b40_ctx cell must NOT be caught by the b4 prefix.
    assert not "b40_ctx1024".startswith("b4_")


@pytest.mark.parametrize("tag,expect_nofire", [
    ("DECODE_ROUTE_NOFIRE", True), ("GRAPH_ROUTE_DEAD", True),
    ("BATCHED_GRAPH_ROUTE_DEAD", True), ("SOMETHING_NEW_DEAD", True),
    ("DECODE_BF16_UNROUTED", False), ("PACKED_ONLY_SPEC", False),
])
def test_suffix_classifier(tag, expect_nofire):
    """DECODE_BF16_UNROUTED is deliberately NOT a nofire tag -- it is per-batch, so it must
    not demote an arm whose other batch sizes routed correctly."""
    assert _bench()._is_nofire_tag(tag) is expect_nofire
