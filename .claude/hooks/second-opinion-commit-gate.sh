#!/usr/bin/env bash
# Claude Code PreToolUse hook (matcher: Bash).
# When Claude is about to run `git commit`, runs `codex review --commit <SHA>`
# on a virtual commit representing the staged diff (created via
# `git commit-tree`, never touches HEAD or any ref) and surfaces findings
# to Claude. Provider: Azure-hosted gpt-5.5 via the `azure` profile in
# ~/.codex/config.toml; reachable endpoint pinned at `oaidr5` after
# 2026-05-06 testing showed the default `t2vgoaigpt4o3` priority queue
# returns `too_many_requests` while `oaidr5` is healthy and ~2× faster.
#
# History: this hook used to run a 3-specialist Copilot/GPT-5.5 review.
# That worked but had two problems:
#   1. ARG_MAX trip on diffs > ~150 KB (the prompt+diff was passed as a
#      CLI arg to `copilot -p`), which caused the hook to silently fail
#      to-advisory and the commit proceeded without a real review.
#      Found 2026-05-06 while debugging the pipeline-b-env-hardening
#      apply commit (4557 lines / 230 KB) committing clean with no
#      visible findings.
#   2. Coupled to one provider; can't easily migrate to other projects
#      that don't have Copilot installed but DO have codex + Azure.
# Codex review --commit avoids both: the diff is referenced via SHA
# (no ARG_MAX), the prompt is built-in (consistent across projects),
# and `codex review` is the standard non-interactive review entry
# point in codex CLI 0.122+.
#
# Mode (default: BLOCKING) — on any [P0] finding the hook exits 2, which
#   makes Claude Code show stderr to the model AND block the commit.
#   Claude must address the finding (or add [skip-review] to the commit
#   message) before it can retry. P1 findings remain advisory (exit 1,
#   user-visible only). Set SECOND_OPINION_ADVISORY=1 to disable blocking
#   (P0 → exit 1 like P1).
#
# Latency: ~60–180 s per commit (single codex review pass; the diff is
#   read by the model's Read tool in chunks, so big diffs scale roughly
#   linearly with file count rather than total bytes).
#
# Skip rules (silently exit 0 when any match):
#   - codex CLI not on PATH (graceful no-op)
#   - Azure profile not configured in ~/.codex/config.toml (graceful no-op)
#   - SKIP_SECOND_OPINION=1 in env
#   - commit message contains [skip-review] or [no-verify]
#   - rebase / cherry-pick / merge in progress
#   - branch matches wip/* explore/* propose/*
#   - empty diff (or --amend with empty staged)
#   - all changed paths in ignore set (*.md, *.lock, data/, cache/, results/)
#
# Migration to other projects: copy this file + ensure codex CLI 0.122+
# is on PATH + the [profiles.azure] block in ~/.codex/config.toml points
# at a managed-identity-authenticated Azure OpenAI endpoint that hosts
# gpt-5.5. No project-specific values; the hook is portable.

set -uo pipefail

# Graceful no-op if `codex` CLI not installed or not on PATH.
# (Allows collaborators without codex CLI to clone+commit without errors.)
command -v codex >/dev/null 2>&1 || exit 0

# Read hook payload (Claude Code passes JSON on stdin)
PAYLOAD=$(cat)
COMMAND=$(printf '%s' "$PAYLOAD" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("tool_input", {}).get("command", ""), end="")
except Exception:
    pass
' 2>/dev/null) || COMMAND=""

# ============================================================
# Trigger gate: only act on `git commit` invocations
# ============================================================

# Match `git commit` at start-of-command or after `&&` / `;`
if ! printf '%s' "$COMMAND" | grep -qE '(^|&&[[:space:]]*|;[[:space:]]*)git[[:space:]]+commit\b'; then
    exit 0
fi

# ============================================================
# Skip rules
# ============================================================

# 0. env override
[[ "${SKIP_SECOND_OPINION:-0}" == "1" ]] && exit 0

# 1. skip markers in commit message
if printf '%s' "$COMMAND" | grep -qE '\[(skip-review|no-verify)\]'; then
    exit 0
fi

# 2. ensure we're in a git repo; if not, allow silently.
#
# PreToolUse hooks fire BEFORE the wrapped command runs, so any leading
# ``cd <path> && git commit ...`` in the COMMAND has not yet taken
# effect — the hook's CWD is the parent process's CWD, which may be a
# completely different repo. To find the *intended* repo we parse the
# leading ``cd <path>`` prefix (if any) from the command and chdir
# there; otherwise fall back to the current CWD. Without this, the
# hook silently runs ``git diff --cached`` in the wrong worktree, sees
# an empty diff, and exits 0 — review never fires.
WORKDIR=$(printf '%s' "$COMMAND" \
    | sed -nE 's|^[[:space:]]*cd[[:space:]]+"?([^"&;[:space:]]+)"?[[:space:]]*&&.*|\1|p' \
    | head -1)
if [[ -n "$WORKDIR" && -d "$WORKDIR" ]]; then
    cd "$WORKDIR" || exit 0
fi

GIT_DIR=$(git rev-parse --git-dir 2>/dev/null) || exit 0

# 3. rebase / cherry-pick / merge in progress — don't second-guess history rewrites
for marker in REBASE_HEAD CHERRY_PICK_HEAD MERGE_HEAD; do
    [[ -e "$GIT_DIR/$marker" ]] && exit 0
done
[[ -d "$GIT_DIR/rebase-merge" ]] && exit 0
[[ -d "$GIT_DIR/rebase-apply" ]] && exit 0

# 4. branch is WIP / exploratory
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
case "$BRANCH" in
    wip/*|explore/*|propose/*) exit 0 ;;
esac

# 5. determine diff scope (handle -a / --all)
if printf '%s' "$COMMAND" | grep -qE 'git[[:space:]]+commit\b[^"'"'"']*[[:space:]]-[A-Za-z]*a([A-Za-z]|\b|[[:space:]])' \
   || printf '%s' "$COMMAND" | grep -qE 'git[[:space:]]+commit\b[^"'"'"']*--all\b'; then
    DIFF=$(git diff HEAD 2>/dev/null)
else
    DIFF=$(git diff --cached 2>/dev/null)
fi

# 6. amend with no new staged content
if printf '%s' "$COMMAND" | grep -qE -- '--amend' && [[ -z "$DIFF" ]]; then
    exit 0
fi

# 7. truly empty diff
[[ -z "$DIFF" ]] && exit 0

# 8. diff size cap
DIFF_LINES=$(printf '%s\n' "$DIFF" | wc -l)
if (( DIFF_LINES > 5000 )); then
    echo "[second-opinion: skipped, diff too large ($DIFF_LINES lines); review manually if needed]" >&2
    exit 0
fi

# 9. all changed paths are in ignore set
CHANGED=$(printf '%s\n' "$DIFF" | grep -E '^\+\+\+ ' | sed -e 's|^+++ b/||' -e 's|^+++ a/||' -e 's|^+++ ||' | grep -v '^/dev/null$' || true)
SUBSTANTIVE=$(printf '%s\n' "$CHANGED" | grep -vE '(^|/)(\.gitignore|.*\.md|.*\.lock|uv\.lock)$' | grep -vE '^(data|cache|results)/' || true)
if [[ -z "$SUBSTANTIVE" ]]; then
    exit 0
fi

# ============================================================
# codex review --commit against a virtual commit
# ============================================================
#
# `git commit-tree` writes a commit object referencing the index's
# tree against HEAD as parent. The resulting SHA is dangling — not
# on any ref or branch — so the working tree, the staging area,
# branch HEAD, and the reflog are all untouched. codex review reads
# it like a normal commit. The dangling object gets garbage-collected
# on the next `git gc`; until then it costs ~200 bytes of objects/
# storage per commit attempt, well below the noise floor.
#
# Why a virtual commit instead of `codex review --uncommitted`:
# `--uncommitted` reviews staged + unstaged + untracked, which would
# pollute the review with whatever the operator happens to have in
# their working tree at commit time (e.g., a half-edited unrelated
# file, or this hook script itself if it's mid-edit). The virtual
# commit captures EXACTLY the diff that's about to land and nothing
# else.

TREE_SHA=$(git write-tree 2>/dev/null) || exit 0
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null) || exit 0
COMMIT_SHA=$(git commit-tree "$TREE_SHA" -p "$HEAD_SHA" \
    -m "second-opinion review staging (transient)" 2>/dev/null) || exit 0

echo "[second-opinion: reviewing $DIFF_LINES-line diff via codex review (gpt-5.5 / oaidr5), ~2-10 min depending on size]" >&2

# codex review --commit uses its built-in review prompt (kept in
# sync upstream by openai/codex). We override the model + provider
# via -c so this works on any project that has `[profiles.azure]`
# in ~/.codex/config.toml plus an oaidr5-equivalent endpoint.
#
# `service_tier=flex` matters: the default `fast` (priority queue)
# returns `too_many_requests` reliably under our usage, while `flex`
# (standard queue) is rate-limited far less aggressively. Verified
# 2026-05-06 across t2vgoaigpt4o3 / oaidr5 / oaidr9.
#
# `model_reasoning_effort=high` (NOT xhigh): xhigh produces deeper
# findings but a 4500-line diff at xhigh exceeds 5 minutes wall-clock.
# `high` is the sweet spot — finds the same class of P0/P1 issues
# but completes in 2-4 min for typical commits, 6-10 min for whole-
# feature applies. Operators willing to wait can override via
# `-c model_reasoning_effort='"xhigh"'` env (not currently wired).
#
# `--commit <SHA>` mode rejects custom prompts (mutually exclusive),
# so we live with codex's built-in review prompt — which is fine,
# it's well-tuned for general code review and works across project
# styles.
#
# Timeout: 600s (10 min) covers typical commits + the whole-feature
# apply class (≤ ~5000 lines per the diff_size cap above). On big
# diffs that exceed budget, exit 124 from `timeout` falls through
# to "REVIEW empty → silent exit 0" — which is suboptimal (silent
# bypass) but matches the failure mode of the original Copilot
# variant. Operators wanting hard guarantees on big diffs should
# split the commit.
REVIEW=$(timeout 600 codex review --commit "$COMMIT_SHA" \
    -c sandbox_mode='"read-only"' \
    -c model='"gpt-5.5"' \
    -c model_provider='"azure"' \
    -c model_reasoning_effort='"high"' \
    -c service_tier='"flex"' \
    -c model_providers.azure.base_url='"https://oaidr5.openai.azure.com/openai/v1"' \
    2>&1)
CODEX_RC=$?

# `timeout` returns 124 when it sends SIGTERM. `codex` itself can
# exit non-zero on transport hiccups (Azure 5xx, network blip, OOM
# in the sandbox). In either case REVIEW will likely be partial /
# missing findings. Surface this as ADVISORY (not silent) so the
# operator at least sees "review didn't complete cleanly" instead
# of mistaking a broken gate for a clean LGTM. (This is the failure
# mode that bit the original Copilot hook on 4500-line diffs.)
if (( CODEX_RC != 0 )); then
    {
        echo ""
        echo "=== second-opinion advisory: codex review exited rc=$CODEX_RC (timeout=600s, big diff?) ==="
        echo "Last 30 lines of codex output:"
        printf '%s\n' "$REVIEW" | tail -30
        echo "=== suppress: [skip-review] in msg, SKIP_SECOND_OPINION=1 ==="
        echo ""
    } >&2
    # Advisory exit 1 — commit proceeds. Operator decides whether
    # to investigate manually before pushing.
    exit 1
fi

# Parse the structured findings out of codex's output. Codex review's
# built-in prompt emits findings as Markdown bullet items with a
# severity tag prefix:
#     - [P0] <title> — <path>:<line>
#     - [P1] <other>
#     - [P2] <minor>
# (Free-text summary paragraphs above/below; they're filtered out
# by the regex below.)
FINDINGS=$(printf '%s\n' "$REVIEW" | grep -E '^[[:space:]]*-[[:space:]]+\[P[012]\]' || true)

# No findings = LGTM-equivalent → silent exit 0.
if [[ -z "$FINDINGS" ]]; then
    exit 0
fi

HAS_P0=0
printf '%s\n' "$FINDINGS" | grep -qE '^[[:space:]]*-[[:space:]]+\[P0\]' && HAS_P0=1

# Claude Code hook exit-code contract (per `/hooks` UI):
#   exit 0 — stdout/stderr NOT shown
#   exit 1 — show stderr to USER only, continue with tool call (advisory)
#   exit 2 — show stderr to MODEL AND block tool call           (blocking)
#
# Default: any [P0] → blocking (exit 2). P1/P2-only → advisory (exit 1).
# Set SECOND_OPINION_ADVISORY=1 to disable blocking entirely (always exit 1).
if (( HAS_P0 == 1 )) && [[ "${SECOND_OPINION_ADVISORY:-0}" != "1" ]]; then
    MODE_BANNER="=== second-opinion BLOCKING (P0 found; address or add [skip-review] to retry) ==="
    EXIT_CODE=2
else
    MODE_BANNER="=== second-opinion advisory (commit will proceed) ==="
    EXIT_CODE=1
fi

{
    echo ""
    echo "$MODE_BANNER"
    printf '%s\n' "$FINDINGS"
    echo "=== suppress: [skip-review] in msg, SKIP_SECOND_OPINION=1, or SECOND_OPINION_ADVISORY=1 ==="
    echo ""
} >&2

exit "$EXIT_CODE"
