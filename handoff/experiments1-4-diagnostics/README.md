# Experiments 1–4 diagnostics bundle

This compact archive preserves original server paths for the available project logs, diagnostics, small result summaries, setup records, and reports. The inventory covers 258 source files, including 41 `.log` files. It includes Experiment 2 compare/decompose and replay diagnostics, Experiment 3 baseline/oracle/final verification and core/four logs, Experiment 4 task-transfer records, Hugging Face Xet diagnostics, provisioning logs, and earlier reproduction reports.

The archive intentionally excludes datasets, model weights, adapters/heads, downloaded dataset/model archives, evaluation caches, general caches, and credentials. A credential-marker scan found no credential-like values in the selected sources.

The Experiment 1 official guide and surviving reproduction outputs are included, but no standalone Experiment 1 run log or complete original Experiment 1 result set was found in the scanned project paths.

Verify the archive with:

```bash
sha256sum -c DC-Merge-experiment-diagnostics-20261019.tar.gz.sha256
```

`FILE_LIST.csv` and `SOURCE_INVENTORY.csv` identify source and archive paths, sizes, categories, and hashes. `SHA256SUMS` covers the archive members other than itself; it has been verified against the archive contents.
