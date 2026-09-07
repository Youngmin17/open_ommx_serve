import logging
import hashlib
import os
import re
from functools import cache
from typing import TYPE_CHECKING, Union

from transformers import AutoTokenizer


if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)
_TOKENIZER_LOAD_AUDIT: list[dict] = []

# Upstream ships [4096]. Kept at 32768 deliberately: this is the fallback used only when a
# caller passes no max_seq_lengths, and the KV-quant RULER track relies on the 32K default.
# Changing it would silently move that track's lengths, so the deviation stays and is
# documented here rather than reverted.
DEFAULT_SEQ_LENGTHS = [
    32768,
]


@cache
def _load_tokenizer(pretrained, requested_revision):
    load_kwargs = {"trust_remote_code": True}
    if requested_revision is not None:
        load_kwargs["revision"] = requested_revision
    loaded = AutoTokenizer.from_pretrained(pretrained, **load_kwargs)

    # Transformers does not consistently retain ``_commit_hash`` on tokenizer
    # objects.  Its resolved file paths do retain ``snapshots/<commit>/...``.
    # Formal callers pass an immutable revision; fail closed unless every
    # resolved snapshot path agrees with it, then record hashes of those files.
    resolved_paths: set[str] = set()

    def collect_paths(value):
        if isinstance(value, str) and os.path.isfile(value):
            # Keep the snapshot symlink path: resolving it to ``blobs/<hash>``
            # would discard the commit component that this audit must verify.
            resolved_paths.add(os.path.abspath(value))
        elif isinstance(value, dict):
            for child in value.values():
                collect_paths(child)
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                collect_paths(child)

    collect_paths(getattr(loaded, "init_kwargs", {}))
    for name in ("tokenizer_file", "vocab_file", "merges_file", "sp_model_file"):
        collect_paths(getattr(loaded, name, None))

    snapshot_commits: set[str] = set()
    outside_snapshot_paths: list[str] = []
    for path in resolved_paths:
        match = re.search(r"(?:^|/)snapshots/([0-9a-f]{40})(?:/|$)", path)
        if match:
            snapshot_commits.add(match.group(1))
        else:
            outside_snapshot_paths.append(path)
    if requested_revision is not None and (
        snapshot_commits != {requested_revision} or outside_snapshot_paths
    ):
        raise RuntimeError(
            "RULER tokenizer resolved outside the requested snapshot: "
            f"requested={requested_revision} resolved={sorted(snapshot_commits)} "
            f"outside={sorted(outside_snapshot_paths)}"
        )

    resolved_hashes = {}
    for path in sorted(resolved_paths):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        resolved_hashes[path] = digest.hexdigest()
    evidence = {
        "repo_id": str(pretrained),
        "requested_revision": requested_revision,
        "effective_revision": (
            requested_revision
            if snapshot_commits == {requested_revision} and not outside_snapshot_paths
            else None
        ),
        "snapshot_commits": sorted(snapshot_commits),
        "outside_snapshot_paths": sorted(outside_snapshot_paths),
        "resolved_file_sha256": resolved_hashes,
    }
    return loaded, evidence


def get_tokenizer(
    tokenizer=None, pretrained=None, tokenizer_revision=None, revision=None,
    ruler_builder=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    # UPSTREAM assert, restored. The local replacement silently fell back to
    # "togethercomputer/Llama-2-7B-32K" whenever no tokenizer reached this function, so a
    # run could synthesise its "16K" prompts with a Llama-2 tokenizer while evaluating a
    # Qwen model -- every length, and therefore every sparsity target, off by the
    # tokeniser ratio, with only a warning in the log. Failing loudly is correct: a wrong
    # tokenizer invalidates the run, it does not degrade it.
    assert pretrained, "No tokenizer or pretrained provided."

    if tokenizer_revision and revision and tokenizer_revision != revision:
        raise ValueError(
            "conflicting RULER tokenizer revisions: "
            f"{tokenizer_revision!r} != {revision!r}"
        )
    requested_revision = tokenizer_revision or revision
    if requested_revision is not None and not re.fullmatch(
        r"[0-9a-f]{40}", requested_revision
    ):
        raise ValueError(
            "RULER tokenizer_revision must be an immutable lowercase 40-hex commit"
        )

    eval_logger.info(
        "Using tokenizer %s@%s for synthetic task builder %s.",
        pretrained,
        requested_revision or "default",
        ruler_builder or "unspecified",
    )
    loaded, evidence = _load_tokenizer(str(pretrained), requested_revision)
    _TOKENIZER_LOAD_AUDIT.append({**evidence, "builder": ruler_builder})
    return loaded


def get_tokenizer_load_audit() -> list[dict]:
    """Return immutable-revision evidence for RULER synthetic-data tokenizers."""

    return [
        {
            **record,
            "snapshot_commits": list(record["snapshot_commits"]),
            "outside_snapshot_paths": list(record["outside_snapshot_paths"]),
            "resolved_file_sha256": dict(record["resolved_file_sha256"]),
        }
        for record in _TOKENIZER_LOAD_AUDIT
    ]


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_all(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    # hacky: set all other lengths to -1
    metrics = {str(length): -1.0 for length in DEFAULT_SEQ_LENGTHS}
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_part(pred, [doc["outputs"]])
    metrics[str(input_len)] = score
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        # we don't have any samples with this length
        return -1
    return sum(res) / len(res)
