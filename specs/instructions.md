# KDA PR Refactor Instructions

## Scope and branch

- Work only in this `kda-pr` worktree.
- Use local branch `momcilo/kda-pr-refactor`, based on
  `origin/momcilo/feature/kda-pr`.
- Implement the full development specification in `kda-development.md`.
- Treat `kda-layer-redesign.md` as the approved design and
  `kda-layer-redesign-evidence.md` as supporting evidence.

## Execution

- Begin only on the user's initial “go.”
- After that initial authorization, work autonomously; do not wait for further
  user input for in-scope work.
- Record and act on in-scope decisions. Escalate only when work requires new
  authority outside the approved specification.

## Ledger and commits

- Maintain `kda-pr-refactor-work-ledger.md` in this directory.
- Add concise UTC-timestamped entries with emoji status markers.
- Record progress, problems, failures, resolutions, and decisions.
- Validate each completed concern and create incremental local commits.

## Remote-state boundary

- Do not push upstream, open a pull request, rewrite history, or otherwise
  change remote state.
- When implementation is complete, stop for the user's review of the ledger
  and local commits before any push.
