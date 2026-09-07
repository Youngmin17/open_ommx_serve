#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired analysis of the LiveCodeBench-v6 KV-quant arms, and the figure's data file.

Two things make a naive pass@1 table misleading here, so this script never prints one alone.

1. Arms cover different id sets whenever a run is stopped early, and the seeded subset is not
   uniform in difficulty, so raw per-arm means are not comparable to each other. Two views fix
   this, and the primary one is the intersection:

     COMMON   every arm, including bf16, restricted to the ids EVERY arm generated. One problem
              set for all arms, so arm-vs-arm ranking is licensed. n is the smallest arm's
              coverage; that is the price of the guarantee.
     PER-ARM  each arm on all of its own ids, paired against bf16 restricted to those same ids.
              Uses every generation, but licenses only arm-vs-bf16, never arm-vs-arm.

2. At max_new_tokens=16384 a thinking model runs out of tokens mid-reasoning and never emits its
   code fence, which the benchmark scores 0. Measured: `no_code_block` is 97-100% token-capped and
   100% fence-less, and quantization inflates generation length (kivi +46%, kitty +37% vs bf16),
   so pass@1 partly measures verbosity. Every arm is reported with cap-hit% and with the
   comparison restricted to pairs where neither side hit the cap.

   The cap-free restriction conditions on a post-treatment variable, so it is a bound, not a
   causal estimate. It answers "does this method get answers wrong, or only run out of room?" --
   McNemar on the discordant pairs is the test that question actually needs.

Usage:
  python eval/lcb/analyze_lcb.py --runs <dir> --ids <lcb_subset_ids.json> [--json figure/data/lcb_qwen3_30b.json]
"""
import argparse
import collections
import glob
import json
import math
import os

CAP_TOKENS = 16384
ARMS = ("none", "kivi", "turboquant", "kitty", "ommx")
LABEL = {"none": "bf16", "kivi": "KIVI", "turboquant": "TurboQuant",
         "kitty": "Kitty", "ommx": "OMMX"}
# Average KV bits, PAYLOAD ONLY, for the exact configs run here (see bit_budget.py).
# Scale/zero-point and OMMX's outlier-membership encoding are NOT included -- the comparison is
# not iso-bit and the metadata accounting is not established. Do not read these as a ranking.
BITS = {"none": 16.0, "kivi": 2.000, "turboquant": 3.000, "kitty": 2.328, "ommx": 2.188}
RECIPE = {"none": "bf16, no KV quant",
          "kivi": "K/V INT2, group 128",
          "turboquant": "Walsh-Hadamard rotate + K/V INT3, group 128",
          "kitty": "K/V INT2 + 3/128 bf16 channels, group 128",
          "ommx": "INT2 base + 12/64 FP4 K-outliers, group 64, sink 8, pow2"}


def mcnemar_exact(b10, b01):
    """Two-sided exact binomial test on the discordant pairs."""
    n = b10 + b01
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b10, b01) + 1))
    return min(1.0, 2 * tail / 2 ** n)


# Each method's fp16 region, mirrored from kv_fakequant/quantizers.py PUBLISHED_RECIPE. The cache
# stamps what it actually enforced into kv_desc, so comparing the two catches a run that silently
# dropped a baseline's residual or sink -- the failure that made KIVI and Kitty look worse than
# their papers for an entire campaign.
PUBLISHED_REGION = {"kivi": {"sink": 0, "res": 128},
                    "kitty": {"sink": 32, "res": 0},
                    "turboquant": {"sink": 0, "res": 0},
                    "ommx": {"sink": 8}}


def load_scored(runs, arm, keep):
    p = os.path.join(runs, "livecodebench_v6__%s.json" % arm)
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in json.load(open(p))["results"] if r["id"] in keep}


def check_recipe(runs, arm):
    """-> (ok, message). Reads kv_desc as the run recorded it; None when the run predates it."""
    want = PUBLISHED_REGION.get(arm)
    if not want:
        return True, ""
    for name in ("livecodebench_v6__%s.json" % arm,) + tuple(
            "livecodebench_v6__%s.shard%d.json" % (arm, i) for i in range(4)):
        p = os.path.join(runs, name)
        if not os.path.exists(p):
            continue
        desc = json.load(open(p)).get("kv_desc")
        if not desc:
            continue
        missing = [k for k, v in want.items() if ("%s=%d" % (k, v)) not in desc]
        if missing:
            return False, "kv_desc=%r does not show %s" % (
                desc, ", ".join("%s=%d" % (k, want[k]) for k in missing))
        return True, desc
    return None, "no kv_desc recorded (run predates recipe stamping)"


def load_lengths(runs, arm, keep, tokenizer):
    """Token length per generation; needed to tell "ran out of tokens" from "wrong answer"."""
    out = {}
    for f in sorted(glob.glob(os.path.join(runs, "livecodebench_v6__%s.shard*.gens.jsonl" % arm))):
        for line in open(f):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d["id"] in keep and d["id"] not in out:
                out[d["id"]] = len(tokenizer(d.get("gen", ""), add_special_tokens=False)["input_ids"])
    return out


def summarize(arm, ids, S, base, capped, base_cap, lens):
    """One row: the arm's numbers over exactly `ids`, paired against bf16 on those same ids."""
    free = [i for i in ids if not capped[arm].get(i, True) and not base_cap.get(i, True)]
    b10 = sum(1 for i in ids if base[i]["score"] and not S[i]["score"])
    b01 = sum(1 for i in ids if not base[i]["score"] and S[i]["score"])
    f10 = sum(1 for i in free if base[i]["score"] and not S[i]["score"])
    f01 = sum(1 for i in free if not base[i]["score"] and S[i]["score"])
    L = sorted(lens[arm][i] for i in ids)
    base_len = sum(lens["none"][i] for i in ids) / len(ids)
    return dict(
        arm=arm, label=LABEL[arm], recipe=RECIPE[arm], bits=BITS[arm], n=len(ids),
        pass1=sum(S[i]["score"] for i in ids) / len(ids),
        bf16=sum(base[i]["score"] for i in ids) / len(ids),
        cap_hit=sum(capped[arm].get(i, False) for i in ids) / len(ids),
        n_free=len(free),
        pass1_free=(sum(S[i]["score"] for i in free) / len(free)) if free else None,
        bf16_free=(sum(base[i]["score"] for i in free) / len(free)) if free else None,
        mcnemar={"bf16_only": b10, "arm_only": b01, "p": mcnemar_exact(b10, b01)},
        mcnemar_capfree={"bf16_only": f10, "arm_only": f01, "p": mcnemar_exact(f10, f01)},
        median_tokens=L[len(L) // 2],
        length_vs_bf16=(sum(L) / len(L) - base_len) / base_len,
        status=dict(collections.Counter(S[i].get("status") for i in ids).most_common()),
    )


def print_table(rows, title, note):
    w = "%-12s %5s %8s %8s %8s %9s %9s %9s  %s"
    print("\n%s" % title)
    print("  %s" % note)
    print(w % ("arm", "n", "pass@1", "bf16", "delta", "cap-hit%", "no-cap", "bf16|nc", "KV bits"))
    print("-" * 92)
    for r in rows:
        print(w % (r["label"], r["n"], "%.4f" % r["pass1"], "%.4f" % r["bf16"],
                   "%+.4f" % (r["pass1"] - r["bf16"]), "%.1f" % (100 * r["cap_hit"]),
                   "%.4f" % (r["pass1_free"] or 0), "%.4f" % (r["bf16_free"] or 0),
                   "%.3f" % r["bits"]))
    for label, key, nk in (("ALL pairs", "mcnemar", "n"),
                           ("CAP-FREE pairs", "mcnemar_capfree", "n_free")):
        print("\n  McNemar vs bf16 -- %s" % label)
        print("    %-12s %5s %9s %9s %10s  %s"
              % ("arm", "n", "bf16 only", "arm only", "p", "verdict"))
        for r in rows:
            if r["arm"] == "none":
                continue
            m = r[key]
            v = ("worse ***" if m["p"] < 0.05 and m["bf16_only"] > m["arm_only"]
                 else "tie (0 discordant)" if m["bf16_only"] == m["arm_only"] == 0 else "n.s.")
            print("    %-12s %5d %9d %9d %10.4g  %s"
                  % (r["label"], r[nk], m["bf16_only"], m["arm_only"], m["p"], v))


def main(a):
    keep = set(json.load(open(a.ids))["120"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    scored = {arm: load_scored(a.runs, arm, keep) for arm in ARMS}
    scored = {arm: S for arm, S in scored.items() if S}
    lens = {arm: load_lengths(a.runs, arm, keep, tok) for arm in scored}
    capped = {arm: {i: n >= CAP_TOKENS - 2 for i, n in L.items()} for arm, L in lens.items()}
    if "none" not in scored:
        raise SystemExit("no bf16 baseline in %s -- score arm `none` first" % a.runs)
    base, base_cap = scored["none"], capped["none"]

    # COMMON: the ids every arm generated. Equalising n is what buys arm-vs-arm comparability;
    # an arm stopped early otherwise sits on an easier or harder slice of the subset.
    common = set(keep)
    for arm, S in scored.items():
        common &= set(S) & set(lens[arm])
    common = [i for i in json.load(open(a.ids))["120"] if i in common]

    order = [x for x in ARMS if x in scored]
    rows_common = [summarize(arm, common, scored[arm], base, capped, base_cap, lens)
                   for arm in order]
    rows_own = [summarize(arm, [i for i in scored[arm] if i in base], scored[arm], base,
                          capped, base_cap, lens) for arm in order]
    rows_common.sort(key=lambda r: -r["pass1"])
    rows_own.sort(key=lambda r: -r["pass1"])

    print("recipe fidelity -- does the recorded kv_desc show each method's published fp16 region?")
    unverified = []
    for arm in order:
        ok, msg = check_recipe(a.runs, arm)
        mark = {True: "ok", False: "MISMATCH", None: "unverified"}[ok]
        print("  %-12s %-11s %s" % (LABEL[arm], mark, msg))
        if ok is not True and arm != "none":
            unverified.append(arm)
    if unverified:
        print("  !! %s ran without a verified fp16 region. Their columns bound those methods from"
              % ", ".join(LABEL[x] for x in unverified))
        print("     below; they do not reproduce them. See eval/lcb/README.md.")
    print()

    print("LiveCodeBench-v6 (release_v6 union, seeded n=120 subset, subset_seed=20250729)")
    print("pass@1 @ 1 stochastic sample, T=0.6 top_p=0.95 top_k=20 seed=42, batch_size=1, "
          "max_new_tokens=%d" % CAP_TOKENS)
    print("cache-level KV fake-quant; unofficial harness -- not comparable to published LCB numbers")

    print_table(rows_common, "PRIMARY -- COMMON SUBSET (all arms on identical problems)",
                "n=%d ids generated by every arm, out of the %d-id subset. Arm-vs-arm ranking is "
                "licensed here." % (len(common), len(keep)))
    print_table(rows_own, "SECONDARY -- PER-ARM COVERAGE (each arm on all of its own ids)",
                "uses every generation, but arms sit on different problem sets: compare each arm "
                "to its own bf16 column only, never to another arm.")

    print("\ngeneration length on the common subset "
          "(a quantized model that rambles hits the cap and scores 0)")
    for r in rows_common:
        print("  %-12s median=%6d tok   vs bf16 %+6.1f%%   cap-hit %4.1f%%   [%s]"
              % (r["label"], r["median_tokens"], 100 * r["length_vs_bf16"],
                 100 * r["cap_hit"], r["recipe"]))

    print("\narm vs arm on the common subset (paired McNemar, n=%d)" % len(common))
    quant = [r for r in rows_common if r["arm"] != "none"]
    for i, ra in enumerate(rows_common):
        for rb in rows_common[i + 1:]:
            A, B = scored[ra["arm"]], scored[rb["arm"]]
            ab = sum(1 for x in common if A[x]["score"] and not B[x]["score"])
            ba = sum(1 for x in common if not A[x]["score"] and B[x]["score"])
            p = mcnemar_exact(ab, ba)
            print("  %-11s vs %-11s  %s only=%2d  %s only=%2d  p=%-10.4g %s"
                  % (ra["label"], rb["label"], ra["label"], ab, rb["label"], ba, p,
                     "***" if p < 0.05 else "n.s."))
    rows = rows_common

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"benchmark": "livecodebench_v6", "subset_seed": 20250729, "subset_n": 120,
                   "model": a.model, "cap_tokens": CAP_TOKENS, "batch_size": 1,
                   "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 42,
                   "harness": "unofficial", "common_n": len(common),
                   "arms": rows_common, "arms_own_coverage": rows_own},
                  open(a.json, "w"), indent=2)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="directory holding livecodebench_v6__*.json")
    p.add_argument("--ids", required=True, help="lcb_subset_ids.json (canonical seeded id lists)")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    p.add_argument("--model", default="Qwen3-30B-A3B-Thinking-2507")
    p.add_argument("--json", default=None, help="also write the figure's data file")
    main(p.parse_args())
