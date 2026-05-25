# Confidentiality Audit: jpeg-rusturbo

**実施日**: 2026-05-26
**Mode**: scope=commit
**監査対象**: staged tree (`BENCH.md` のみ) + `_uchgs-commit-message.txt`

---

## Summary

| カテゴリ                            | Critical | Warning | Info | Pass |
| ----------------------------------- | -------: | ------: | ---: | ---: |
| Phase / Step / Tier / Block         |        0 |       0 |    0 |    ✓ |
| Hostname / IP                       |        0 |       0 |    0 |    ✓ |
| Path (env-specific)                 |        0 |       0 |    0 |    ✓ |
| Production references               |        0 |       0 |    0 |    ✓ |
| Service / customer names            |        0 |       0 |    0 |    ✓ |
| Code names                          |        0 |       0 |    0 |    ✓ |
| Commit message (commit mode)        |        0 |       0 |    0 |    ✓ |

**Total findings: 0**

---

## Mechanical layer (Phase 2 / 3) results

- Internal label grep (`Phase \d+ / Step / Tier / Block / production`)
  on staged diff + commit message: **no matches**.
- Internal label grep also covered ad-hoc patterns `H[0-9]` / `S[0-9]`
  in the commit message (defensive — orchestrator briefing used H5/S4
  IDs internally and the user's no-internal-label rule must hold): **no matches**.
- Path-like token grep on staged diff: only `src/bin/bench.rs`
  (repo-relative source-tree reference, not an env-specific path).
- ast-grep comment extraction: BENCH.md is markdown without
  code-comment nodes; skipped (`sg --lang markdown` not applicable
  for ``$$$COMMENT``).

## LLM layer (Phase 4) judgment

Read the staged diff (BENCH.md, +57 / −23 lines) and the commit
message in full. Findings by category:

- **CPU / host identifiers**: `Xeon Platinum 8272CL`, `Apple M-series`,
  `Cascade Lake`. All are public product / micro-architecture names
  that already appeared in the pre-existing BENCH.md hosts table —
  no change in disclosure surface.
- **Region name**: `Azure centralus` is referenced in the header text
  (pre-existing in the file from v0.6.0) — public cloud region, safe.
- **File / symbol references in prose**: `tests/decode_x86_64.rs`,
  `tests/decode_neon.rs`, `src/bin/bench.rs`, `decode_ac_fast`,
  `decode_dc_fast`, `arch::backend::dct`. All are repo-internal
  symbols / relative paths, intentional cross-refs for readers.
- **Numerical content**: bench timings + ratios + measured gains.
  No leak surface.
- **Commit message**: technical narrative describing measurement
  outcome (`+10-16% sparse natural`, `noise-floor combined LUT`,
  `0.76× vs image`). No internal labels (no Phase / Step / Tier /
  Block / H?-S?), no env-specific paths, no host identifiers
  beyond the public Cascade Lake name.

No path-like token contained a real username, personal directory,
or internal mount point. No hostname matched the `[a-z]+\d+\.(corp|
internal|local)` pattern. No customer / service / org name appeared.

---

## Findings

None.

---

## Disposition

0 findings → receipt issued with `--fix-needed=false` per umbrella
gate-pass rule and `common.md` severity-agnostic policy.
