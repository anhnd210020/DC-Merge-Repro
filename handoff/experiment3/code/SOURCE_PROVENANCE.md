# Source provenance

The Experiment 3 runner files and the matching `vision_lora_merge` implementation were copied from `/workspace/ducanh/DC-Merge`, the source tree used for the completed run. That source tree had no Git metadata available at handoff time, so the exact local runner revision is recorded by the file hashes in the package-root [`SHA256SUMS`](../SHA256SUMS) rather than a local Git commit ID.

`lane_harm/OFFICIAL_COMMIT.txt` records the upstream DC-Merge commit used for the official operator. `environment_vision.yml` is the source package’s environment recipe; it is not a complete lockfile for every installed transitive dependency. Result records, search caches, and logs are copied separately under `results/`.

The package excludes the base model, adapters, classifier heads, dataset images, authentication files, and the full multi-megabyte progress logs. The small shuffled-index tensors required by the official loader are included with `vision_lora_merge`.
