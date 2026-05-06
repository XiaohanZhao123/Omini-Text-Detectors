# Claude Code Hooks

Hooks fire automatically when Claude Code's agent performs tool calls in this
repository. Project-shared registration lives in [`../settings.json`](../settings.json);
personal overrides (gitignored) live in `../settings.local.json`.

## `second-opinion-commit-gate.sh`

A **PreToolUse / Bash** hook that intercepts `git commit` invocations from the
Claude Code agent and runs `codex review --commit <SHA>` against a virtual
commit (built via `git commit-tree`, never touches HEAD or any ref) as a
*second-opinion* reviewer. Provider: Azure-hosted **gpt-5.5** via the
`[profiles.azure]` block in `~/.codex/config.toml`. The point is single-model-
bias defeat: the model writing the code is not the model reviewing it.

### Why this exists

When the agent writing the code is also reviewing it, blind spots are shared.
We want a structurally independent second pair of eyes on every commit, fast
enough to fit inside the commit flow.

### Architecture

`codex review --commit <sha>` runs codex's built-in review prompt (kept in
sync upstream by openai/codex) against a virtual commit object representing
the about-to-land staged diff:

```
git write-tree            ← staged content as a tree object
git commit-tree -p HEAD   ← dangling commit object (no ref, no branch update)
codex review --commit <sha>
```

The virtual commit has no impact on branch state — git GC eventually reaps
it. This means `codex` sees a real commit it can `git diff` and `git blame`
against the parent, which is what its built-in prompt was designed for.

Findings are extracted from codex output as Markdown bullet items:
`- [P0] file:line — short sentence` (or `[P1]` / `[P2]`). Free-text summary
paragraphs around the bullets are filtered out.

### History (and why we left Copilot)

The hook used to run a 3-specialist Copilot/GPT-5.5 review (correctness /
concurrency / contract). That worked but had two unfixable problems found
2026-05-06 while debugging why a 4557-line apply commit landed clean with
no visible findings:

1. **ARG_MAX trip on diffs > ~150 KB.** The hook concatenated the entire diff
   into the `copilot -p "${prompt}${DIFF}"` argument; `execve(2)` rejected it
   silently and the hook fell to the malformed-output advisory branch — the
   commit proceeded without an actual review.
2. **Coupled to one provider.** Other projects on the team don't have
   Copilot installed but DO have codex CLI 0.122+ + Azure OpenAI managed-
   identity auth.

`codex review --commit <SHA>` solves both: the diff is referenced by SHA
(no ARG_MAX), the prompt is built-in (consistent across projects), and the
provider is whatever the project has configured.

The first defense round on the very commit that switched providers
(`fix(...): codex-review round-2 P1+P2 findings`) caught a real
`with ThreadPoolExecutor` join-on-exit bug that defeated the design's
wall-clock timeout — a class of bug the previous 4-round review chain
(Anthropic Opus × 2, Copilot 3-specialist × 1) had missed. Vindicated the
switch.

### Behavior

- **Mode (default: BLOCKING on P0)** — any `[P0]` finding makes the hook
  exit 2, which Claude Code interprets as "show stderr to the model **and**
  block the tool call." Claude has to address the finding (typically by
  editing the code and retrying) or add `[skip-review]` / `[no-verify]` to
  the commit message before the commit is allowed through. `P1` / `P2` and
  malformed-output paths remain advisory (exit 1, surfaced to the user but
  don't block).
- **Why default-block** — without blocking, Claude never sees findings
  (exit 1 routes stderr to user only), and the second-opinion is wasted on
  every PR the user doesn't read in real time. Blocking is what makes
  anti-bias work.
- **Opt out**: `export SECOND_OPINION_ADVISORY=1` to disable blocking
  globally (P0 → exit 1, same as P1). Per-commit opt-out via
  `[skip-review]` / `[no-verify]` in the commit message.
- **Latency**: ~60-180 s per typical commit (≤500 lines), 6-10 min for
  whole-feature applies (≤5000 lines). The model reasoning effort is
  pinned at `high` (not `xhigh`) for that reason — `xhigh` doubles the
  wall-clock and finds the same class of P0/P1 issues. Operators willing
  to wait can override locally.
- **Silent on LGTM**: nothing printed when codex finds no `[P0]`/`[P1]`/`[P2]`
  bullet items. The commit proceeds without feedback noise.

### Skip rules (the hook silently exits 0 when any of these match)

| Trigger | Reason |
|---|---|
| `command -v codex` fails | codex CLI not installed; degrades gracefully |
| `SKIP_SECOND_OPINION=1` env | Manual override |
| Commit message contains `[skip-review]` or `[no-verify]` | Inline override |
| Rebase / cherry-pick / merge in progress | Don't second-guess history rewrites |
| Branch matches `wip/*` `explore/*` `propose/*` | Work-in-progress branches |
| Empty staged diff (or empty `--amend`) | Nothing to review |
| Diff > 5000 lines | Run `codex review --commit` manually after splitting |
| All changed paths are `.md` / `.lock` / `data/` / `cache/` / `results/` | Non-code |

### Dependencies

- [`codex` CLI](https://github.com/openai/codex) `>=0.122` on PATH.
- `~/.codex/config.toml` with a `[profiles.azure]` block pointing at a
  managed-identity-authenticated Azure OpenAI endpoint hosting `gpt-5.5`.
  The hook overrides `model_providers.azure.base_url` to `oaidr5` because
  the default `t2vgoaigpt4o3` priority queue returns `too_many_requests`
  for `gpt-5.5` (verified 2026-05-06; `oaidr5` is healthy and ~2× faster).
- `python3` (stdlib only — used to parse hook payload JSON without `jq`).
- `git` ≥ 2.20 (for `git commit-tree -p` semantics + `--commit` SHA support).

### How to use this in another repo

1. Copy `.claude/hooks/second-opinion-commit-gate.sh` into your `.claude/hooks/`.
2. Either commit `.claude/settings.json` with the same `PreToolUse / Bash`
   registration (so collaborators auto-get it), or paste the registration
   into your personal `.claude/settings.local.json` (gitignored, only
   affects you).
3. Ensure each contributor who wants the review has `codex` CLI installed
   and `~/.codex/config.toml` set up with the `[profiles.azure]` block.
   Without it, the hook silently no-ops — no error.

The script is portable: no project-specific values, no hard-coded paths.

### Disabling

| Want | How |
|---|---|
| One commit, full skip | `SKIP_SECOND_OPINION=1 git commit -m "..."` |
| One commit, in-message marker | `git commit -m "fix: typo [skip-review]"` |
| Personal: never block, only warn | `export SECOND_OPINION_ADVISORY=1` (in `~/.bashrc`) |
| Personal: disable entirely | `export SKIP_SECOND_OPINION=1` (in `~/.bashrc`) |
| Repo-wide: turn off for everyone | Remove the `PreToolUse` entry from `.claude/settings.json` |
