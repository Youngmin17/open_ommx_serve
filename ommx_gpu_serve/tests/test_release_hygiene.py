#!/usr/bin/env python3
"""End-to-end test for scripts/check_release_hygiene.sh.

Two of that script's rules shipped dead and nobody noticed, because a gate that
never fires looks exactly like a clean repository:

  * the site-path rule passed ``--perl-regexp`` where git wants ``-P``, so git
    read it as a revision and the rule aborted;
  * the foundry rule asked for ``\\bumc\\b``, which cannot match ``gen_umc_mem``
    at all -- ``_`` is a word character, so there is no boundary there -- and that
    identifier is the exact thing the rule exists to catch.

So this does not check the patterns in isolation. It builds a throwaway git
repository, commits one file per violation, and runs the real script: every rule
must name itself on the file that violates it, and the sanitised version of the
same repository must exit clean. A rule that stops firing fails here.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_release_hygiene.sh"

# (rule name, path, offending content, sanitised content)
CASES = [
    ("weight-blob", "figure/model.safetensors",
     "not really weights, but the extension is the rule\n",
     None),                                   # the fix is deletion, not rewording
    ("credential", "scripts/upload.sh",
     'TOKEN="hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n',
     'TOKEN="${HF_TOKEN:?set HF_TOKEN}"\n'),
    ("site-path", "scripts/run_here.sh",
     "WORK=/scratch/jdoe/nucleus/build\n",
     "WORK=/scratch/<user>/nucleus/build\n"),
    # The rule this repository actually needs: a file whose licence is not the root
     # licence and which no THIRD_PARTY.md row covers.  A directory-level version of
     # this rule flagged all six of ommx_gpu_serve/'s subdirectories -- our own
     # headers say "Copyright" too -- and still walked past the single BSD-3 CUTLASS
     # fork inside our tree, which is the one file that matters.
    ("attribution", "ommx_gpu_serve/vendored.hpp",
     "// SPDX-License-Identifier: BSD-3-Clause\n// forked from somewhere\n",
     "// SPDX-License-Identifier: Apache-2.0\n// ours\n"),
]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _build(repo: Path, sanitised: bool) -> None:
    """Write the fixtures and commit them, so `git ls-files` can see them."""
    for _rule, rel, bad, good in CASES:
        body = good if sanitised else bad
        if body is None:                      # vendor view: absent when sanitised
            continue
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    (repo / "THIRD_PARTY.md").write_text(
        "    OURS_SPDX: Apache-2.0\n    OURS_COPYRIGHT: OMMX Contributors\n"
        "\n| path | licence |\n|---|---|\n", encoding="utf-8")
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    os.chmod(repo / "scripts" / SCRIPT.name, 0o755)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fixture")


def _run(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(repo / "scripts" / SCRIPT.name)],
                          cwd=repo, capture_output=True, text=True)


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script not present")
def test_every_rule_fires_on_its_own_violation():
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        _build(repo, sanitised=False)
        out = _run(repo)
        assert out.returncode != 0, "a repository full of violations must not pass"
        for rule, rel, _bad, _good in CASES:
            assert rule in out.stdout, f"{rule} did not fire\n{out.stdout}"
            assert rel in out.stdout, f"{rule} fired but did not name {rel}\n{out.stdout}"


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script not present")
def test_the_sanitised_repository_passes():
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        _build(repo, sanitised=True)
        out = _run(repo)
        assert out.returncode == 0, (
            "the sanitised fixtures still trip a rule -- that is a false positive, "
            f"and it makes the gate unusable\n{out.stdout}\n{out.stderr}")


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script not present")
def test_an_exemption_is_honoured_and_reviewable():
    """.release-hygiene-allow is tracked on purpose: an exemption is a claim about
    the repository, so it should show up in a diff like any other change."""
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        _build(repo, sanitised=False)
        (repo / ".release-hygiene-allow").write_text(
            "\n".join(f"{rel}  # fixture" for _r, rel, _b, _g in CASES) + "\n",
            encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "allow")
        out = _run(repo)
        assert out.returncode == 0, f"exemptions were not honoured\n{out.stdout}"
