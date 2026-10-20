# Server-only artifact recovery bundle

This folder preserves selected small, non-secret research and recovery artifacts from the Vast instance. It does not contain datasets, model weights, adapters, heads, Hugging Face credentials, partial downloads, or a wholesale source-tree copy. Original server files remain untouched.

- Target branch: `codex/experiment5-signal-check`
- Verified remote branch before this recovery: `13ded01ec4e4d7ba02372f4fd603649e50bb30d3`
- Payload content commit: `e01bba9ef6b3a0d266e2cc295be60f57d63ba61b` (the exact content SHA recorded in `MANIFEST.csv`).
- `remote_file_commit` is the first GitHub commit containing the file bytes; for an already-present duplicate it records the pre-recovery branch commit, and for excluded records it says `not uploaded`. The final branch tip is verified separately because a commit cannot include its own SHA in its contents.

## What was compared

- `/workspace/ducanh/DC-Merge` has 219 regular source files excluding generated `__pycache__`: 201 match the same path and bytes in the handoff, 16 match the same bytes at other handoff paths, and the remaining 2 guide files are exact members of the already-pushed Experiment 1–4 diagnostics archive. No file content from that tree was missing from the verified remote branch.
- The 287-file Experiment 2 archive has 258 exact member-content matches in remote tracked files or embedded archives. Of the 29 remaining members, 28 are generated `.pyc` files. Its one unique source member, `DC-Merge/lane_harm/lanes.py`, is preserved here. The archive itself remains on the server and is not uploaded.
- The original 221-file Experiment 5 ZIP has 219 contents already represented on the remote. Its two unique members are the old `lanes.py` (the same bytes preserved from the E2 archive) and `run_oracle.py` (preserved here). The ZIP itself remains on the server; the user reports an external copy. Its SHA-256 is in `MANIFEST.csv`.
- All 179 regular files under `/srv/dcmerge/results` were checked against remote tracked files and embedded archives: 176 match; the remaining three (pre-uniform cache, E5 `run_check.sh`, and E5 `SHA256SUMS`) are included here.

## Recoverability and excluded items

- The single identified legacy SUN397 image is **not** copied to GitHub or another location. Princeton's [official SUN database page](https://vision.princeton.edu/projects/2010/SUN/) says the images are for research purposes only, but no explicit redistribution permission or license permitting this copy was found. It remains on this server and will be lost if the instance is destroyed. The broader local SUN397 fragment has 36 files (11,335,489 bytes); only the identified 156,160-byte image has a confirmed differing-byte record from the prior inventory, and this turn did not revalidate the other 35 images for uniqueness.
- `/srv/dcmerge/backup_plan/DC-Merge-Repro.bundle` (17,999,385 bytes) remains on the server and is not included. Its refs match remote branch tips except the older Experiment 4 ref, which is an ancestor of the current remote tip; the bundle adds no uniquely reachable commit found in this check. The compact plan, manifests, checksums, environment exports, helper, and the file-level `dataset_contents.csv` inventory are included; that CSV contains paths and hashes only, not dataset bytes.
- `/workspace/.dcmerge_sun397_http_stage` still contains preserved incomplete download files, including the 12,406,746,644-byte Xet partial and three smaller CLIP partials. They are not uploaded or modified.
- The 2026-10-16 Experiment 2 archive itself remains server-only, but its unique source file is preserved here and its other meaningful member contents were found in the remote diagnostics archive.

## Hugging Face recovery pins

See `SOURCE_REVISIONS.md`. Repository access from outside this server has not been tested. No token or credential is included.

`MANIFEST.csv` lists original paths/member paths, sizes, SHA-256, purpose, disposition, the pre-recovery remote commit, and the commit that first introduced the recovery payload. `SHA256SUMS` covers this README, source pins, manifest, and every payload file.
