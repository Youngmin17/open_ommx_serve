#!/usr/bin/env bash
# Release hygiene: the OSS boundary, enforced instead of described.
#
#   scripts/check_release_hygiene.sh            # every tracked file
#   scripts/check_release_hygiene.sh --staged   # only what is about to be committed
#   scripts/check_release_hygiene.sh --policy   # print the rules and exit
#
# This repository has a boundary the NPU half does not: it VENDORS third-party code
# (31k lines across four trees) under a root Apache-2.0 LICENSE that does not cover
# it. Vendoring is redistribution, and a licence file sitting above someone else's
# code does not relicense it. THIRD_PARTY.md is the manifest; this script is the
# part that keeps the manifest true.
#
# The other rules are the same everywhere: a literal credential and a de-anonymised
# site path are the two leaks that cannot be undone after a push, because a rewrite
# does not reach forks, reflogs or anyone who already fetched.
#
# No rule names a host, a user or a machine. Grepping for the real names would put
# the internal list into a public repository -- the exact leak the rule exists to
# prevent -- so they match the SHAPE of a de-anonymised reference instead. A site
# that wants its own names matched puts them in .release-hygiene.local, untracked.
#
# Exemptions live in .release-hygiene-allow, which IS tracked: an exemption is a
# claim about the repository and belongs in review.
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="tracked"
case "${1:-}" in
  --staged) MODE="staged" ;;
  --policy) MODE="policy" ;;
  "") ;;
  *) echo "usage: $0 [--staged|--policy]" >&2; exit 64 ;;
esac

if [ "$MODE" = "policy" ]; then
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

# --- patterns -------------------------------------------------------------------
# Defined once and shared by the rule and by the test, so a rule and its proof
# cannot drift apart. Two rules in the sibling repository shipped dead exactly that
# way, and a gate that never fires looks identical to a clean tree.
PAT_CRED='(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
PAT_SITE='(/scratch/|/home/)(?!<|\$|\{)[A-Za-z][A-Za-z0-9_.-]{2,}/|[A-Za-z0-9_.-]+@[a-z][a-z0-9.-]{2,}:|ssh +[a-z][a-z0-9-]{2,} +[^$<]'
PAT_COPYRIGHT='Copyright \(c\)|Copyright [12][0-9]{3}|SPDX-License-Identifier'

MANIFEST="THIRD_PARTY.md"
ALLOW=".release-hygiene-allow"
LOCAL=".release-hygiene.local"
violations=0

if [ "$MODE" = "staged" ]; then
  FILES=$(git diff --cached --name-only --diff-filter=ACMR)
else
  FILES=$(git ls-files)
fi
[ -n "$FILES" ] || { echo "release hygiene: nothing to check"; exit 0; }

exempt() {
  # The checker is exempt from its own content rules by construction: it has to
  # SPELL the forbidden patterns in order to look for them.
  case "$1" in scripts/check_release_hygiene.sh|*/check_release_hygiene.sh) return 0 ;; esac
  [ -f "$ALLOW" ] || return 1
  local p="$1" line
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(echo "$line" | tr -d '[:space:]')"
    [ -n "$line" ] || continue
    case "$p" in "$line"*) return 0 ;; esac
  done < "$ALLOW"
  return 1
}

report() {
  printf '  %-14s %s:%s\n      %s\n' "$1" "$2" "$3" "$(echo "$4" | cut -c1-100)"
  violations=$((violations + 1))
}

# rule 1 -- literal credentials. Reading a token from the environment is fine.
rule_credential() {
  local hit
  hit=$(git grep -nE "$PAT_CRED" -- $FILES 2>/dev/null | head -20)
  [ -z "$hit" ] || while IFS= read -r ln; do
    local f="${ln%%:*}"; exempt "$f" && continue
    report "credential" "$f" "$(echo "$ln" | cut -d: -f2)" "a literal credential"
  done <<< "$hit"
}

# rule 2 -- de-anonymised infrastructure, matched by shape. A site path is fine when
# the identifying segment is a placeholder or a variable: /scratch/<user>/..., $SCRATCH.
rule_site_path() {
  local hit
  hit=$(git grep -P -n "$PAT_SITE" -- $FILES 2>/dev/null | head -20)
  [ -z "$hit" ] || while IFS= read -r ln; do
    local f="${ln%%:*}"; exempt "$f" && continue
    report "site-path" "$f" "$(echo "$ln" | cut -d: -f2)" "${ln#*:*:}"
  done <<< "$hit"
}

# rule 3 -- model weights and other large blobs. They belong in a model hub, and a
# repository that carries them stops being cloneable for the code.
rule_blob() {
  local f sz
  for f in $FILES; do
    exempt "$f" && continue
    case "$f" in
      *.safetensors|*.bin|*.pt|*.pth|*.ckpt|*.onnx|*.gguf|*.h5|*.npz|*.pkl|*.pickle)
        report "weight-blob" "$f" 0 "a model artifact does not belong in git" ;;
      *) [ -f "$f" ] || continue
         sz=$(wc -c <"$f" 2>/dev/null || echo 0)
         [ "$sz" -gt 10485760 ] && report "large-file" "$f" 0 "$((sz/1048576)) MB tracked" ;;
    esac
  done
}

# rule 4 -- attribution. This is the rule this repository actually needs.
#
#   (a) a tree that carries someone else's copyright but is not declared in the
#       manifest is undeclared redistribution;
#   (b) a declared tree whose licence is still open is a decision nobody has made,
#       and the gate holds it open rather than letting a release close it silently.
rule_attribution() {
  [ -f "$MANIFEST" ] || { report "attribution" "$MANIFEST" 0 "the manifest is missing"; return; }

  # Our own licence and copyright holder come from the manifest, so there is one
  # place to change them and the script does not quietly disagree with the page.
  local ours_spdx ours_holder
  # The manifest indents them as a code block, so ^ alone never matched.
  ours_spdx=$(sed -n 's/^[[:space:]]*OURS_SPDX: *//p' "$MANIFEST" | head -1)
  ours_holder=$(sed -n 's/^[[:space:]]*OURS_COPYRIGHT: *//p' "$MANIFEST" | head -1)
  if [ -z "$ours_spdx" ] || [ -z "$ours_holder" ]; then
    report "attribution" "$MANIFEST" 0 "needs OURS_SPDX: and OURS_COPYRIGHT: lines"
    return
  fi

  # Declared paths, from the table: | `path` | ...
  local declared; declared=$(sed -n 's/^| *`\([^`]*\)`.*/\1/p' "$MANIFEST")

  # FILE level, not directory level. Directory level flagged all of ommx_gpu_serve/
  # because our own headers say "Copyright", and it would still have missed the one
  # file that actually matters: a single BSD-3 CUTLASS fork sitting inside our tree.
  local f spdx other
  for f in $FILES; do
    [ -f "$f" ] || continue
    case "$f" in LICENSE|*/LICENSE|NOTICE|*/NOTICE|"$MANIFEST") continue ;; esac
    exempt "$f" && continue
    # covered by a declared path?
    local covered=0 d
    for d in $declared; do case "$f" in "$d"|"$d"/*) covered=1; break ;; esac; done
    [ "$covered" = 1 ] && continue

    spdx=$(grep -hoE "SPDX-License-Identifier: *[A-Za-z0-9.+-]+" "$f" 2>/dev/null \
           | sed 's/.*: *//' | sort -u | grep -v "^$ours_spdx$" | head -1)
    [ -n "$spdx" ] && report "attribution" "$f" 0 \
      "SPDX says $spdx, not $ours_spdx, and no $MANIFEST row covers it"

    other=$(grep -hE "Copyright" "$f" 2>/dev/null | grep -vF "$ours_holder" | head -1)
    [ -n "$other" ] && [ -z "$spdx" ] && report "attribution" "$f" 0 \
      "third-party copyright with no $MANIFEST row: $(echo "$other" | sed 's/^[^A-Za-z]*//' | cut -c1-60)"
  done

  # A row marked BLOCKER is an open question, and a red gate is what an open
  # question should look like until somebody answers it.
  while IFS= read -r ln; do
    local pth; pth=$(echo "$ln" | sed -n 's/^| *`\([^`]*\)`.*/\1/p')
    [ -n "$pth" ] || continue
    exempt "$pth" && continue
    report "attribution" "$pth" 0 "$MANIFEST marks this BLOCKER (licence unresolved)"
  done < <(grep -E '^\|.*BLOCKER' "$MANIFEST" 2>/dev/null)
}

# rule 5 -- a site's own names, if it chose to supply them. Untracked on purpose.
rule_local_patterns() {
  [ -f "$LOCAL" ] || return 0
  local hit
  hit=$(git grep -nIiEf "$LOCAL" -- $FILES 2>/dev/null | head -20)
  [ -z "$hit" ] || while IFS= read -r ln; do
    local f="${ln%%:*}"; exempt "$f" && continue
    report "local-pattern" "$f" "$(echo "$ln" | cut -d: -f2)" "matches $LOCAL"
  done <<< "$hit"
}

echo "release hygiene ($MODE: $(echo "$FILES" | wc -l | tr -d ' ') files)"
rule_credential
rule_site_path
rule_blob
[ "$MODE" = "staged" ] || rule_attribution   # a whole-tree question, not a per-commit one
rule_local_patterns

if [ "$violations" -eq 0 ]; then
  echo "  OK -- the published boundary holds (LICENSE, THIRD_PARTY.md)"
  exit 0
fi
echo
echo "  $violations violation(s).  Each is one of: remove the artifact, replace the"
echo "  literal with a placeholder or an environment variable, record the upstream"
echo "  in $MANIFEST, or -- if it is genuinely fine -- add the path to $ALLOW with"
echo "  a comment saying why, so the exemption is reviewed like any other change."
exit 1
