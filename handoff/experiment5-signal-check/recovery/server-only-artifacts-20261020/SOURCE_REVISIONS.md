# Immutable source revisions

These pins are read from the saved Experiment 3 baseline audit and the downloader configuration. The first two repositories are private Hugging Face repositories. Access from outside the server was not tested. No credentials are included.

| Purpose | Repository | Immutable revision |
|---|---|---|
| Eight dataset files, including the SUN397 archive | `anhnd210020/DC-Merge-Vision-8Task` | `42fc343338f31c2f41cfa85e97d20e22dfbf9d45` |
| Eight task adapters and classification heads | `anhnd210020/DC-Merge-Vision-B32-r16-8task` | `fcde80f8dccb513997c97e7215ff02a589b2994a` |
| Base CLIP model | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |

Evidence already present on the handoff: `handoff/experiment3/audit/replay-diagnosis.md`; the pinned SUN397 dataset revision and archive SHA/size are also embedded in the preserved single-stream downloader.
