# Nightly brief for gpureg

Each scheduled run follows this file. One run = one small, verified improvement at most.

## 1. Setup
- Repo: `araya906/gpureg`, branch `main`. Attach it with push access, clone it, then
  `pip install --break-system-packages -e ".[test]"`.
- Commit identity: `git config user.name "Álvaro Araya Marín"` and
  `git config user.email "arayamarin98@gmail.com"`.

## 2. Validate
- Run `python validation/run_checks.py` and `pytest -q`.
- Read `validation/results/latest.md`.

## 3. Decide
- **Any check or test fails**: fixing it is this run's only job. Diagnose in `src/gpureg/`.
  Do not loosen a tolerance or change a reference unless you can show the reference is wrong;
  if you do, explain why in the commit message and in `nightly/LOG.md`.
- **Everything passes**: take the first unchecked item in `nightly/ROADMAP.md`. Implement it with
  tests and, where it changes numerical results, a new check in `validation/run_checks.py`.
- If the item is too large for one run, split it in ROADMAP.md and do the first part only.

## 4. Gate
- Re-run `python validation/run_checks.py` and `pytest -q`. Push to `main` **only if both pass**.
- Otherwise push nothing; report what failed and what you tried.

## 5. Record (only when pushing)
- Tick the ROADMAP item; add one dated line to `nightly/LOG.md`
  (keep the last 30 lines; roll older ones into a one-line monthly summary).
- Commit `validation/results/latest.*` and `history.jsonl` with the change.

## Rules
- No `Co-Authored-By` or `Claude-Session` trailers in commits.
- No GitHub Actions, LFS, Codespaces or any paid service.
- Touch only this repository.
- Keep functions under ~50 lines, match the existing style, no unrelated refactors,
  remove anything your change makes unused, no function names duplicated across modules.
- Runs have no GPU: they check correctness and CPU timing only. Do not claim GPU speed-ups.

## Report
End with 3–5 lines: checks status, what changed (or why nothing), next ROADMAP item.
