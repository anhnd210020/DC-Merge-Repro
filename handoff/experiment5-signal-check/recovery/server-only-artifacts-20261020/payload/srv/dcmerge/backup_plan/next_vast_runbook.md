# Faster setup for the next Vast instance

## Evidence from this run

- The Hugging Face Xet download stalled with a 12,406,746,644-byte partial that
  showed no growth during the stall check. Two preserved partial copies remain on
  this instance.
- An authenticated 10 MiB HTTP Range test returned `206` and the correct
  `Content-Range`; the sample rate was 2.95 MiB/s. The later pinned single-stream
  HTTP download of SUN397 sustained about 8.04 MiB/s across a 14,361,296,896-byte
  progress interval and finished with the expected 14,591,037,440-byte size and
  SHA-256. It took about 30 minutes from first progress to the verified log line.
- The full SHA-256 was
  `38afed47c5acce92ac26b36b620e936f4a849e7934e30c65487a17e2dbe0c917` at repository
  revision `42fc343338f31c2f41cfa85e97d20e22dfbf9d45`.
- The laptop-to-server transfer was described as very slow, but no numeric rate is
  in the saved logs. The 64 MiB SFTP test in `BACKUP_PLAN.md` is required before
  estimating that route.

## Storage and copy options

| Option | Best use | Limits to check |
|---|---|---|
| Vast local volume | Reuse datasets/model assets across instances on the same host; it survives instance deletion | Host-bound; cannot move it to a different physical host. Size and mount it at launch. It is separately billed. |
| Vast instance/volume/cloud copy | Move the prepared assets when the next GPU is on a different host; avoids the slow laptop path | Benchmark the actual route. Copy support covers instance, volume, and connected cloud storage endpoints. Verify the destination before removing the source. |
| Pinned resumable HTTP Range | Fetch an authorized dataset source directly to the new instance | Speed and Range support depend on the source and host network. Resume only validated ranges; verify the pinned revision, size, and authoritative digest. |

Vast documents container storage as deleted with the instance and local volumes as
persistent but host-bound. Its volume guide documents `vastai copy` between
instances, volumes, and connected cloud storage. See [storage types](https://vast.ai/docs/guides/instances/storage/types)
and [volumes/copy](https://vast.ai/docs/guides/instances/storage/volumes).

For Hugging Face, the Hub library accepts a full commit hash through `revision`;
`snapshot_download` supports bounded `max_workers`, and the Hub documents Xet's
chunk-based implementation. Use the normal Hub path when preflight is healthy and
fall back to the already proven Range route if Xet stops progressing. See the
[download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)
and [download API](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download).

## Recommended preflight

1. **Choose storage first.** Request a local volume on the intended host and mount
   the project data path there. Confirm it is mounted using `vast-capabilities`
   (`.instance.workspace_is_volume`) and `df -h`. The current instance returned
   `false` for `workspace_is_volume`.
2. **Budget space.** This run's backup payload is 17.879 GiB. Extracted datasets,
   selected model/adapters/heads, and the installed Linux environment add about
   26.8 GB. Keeping both dataset archives and extracted data temporarily pushes the
   working set above 40 GiB. Reserve at least 55–60 GiB of usable storage for the
   environment, staging, logs, and headroom.
3. **Measure each route with small files.** For Hugging Face, fetch a 10–64 MiB
   authenticated range, then compare one stream with four bounded, disjoint
   10–16 MiB ranges. Confirm every response is HTTP `206`, the `Content-Range`
   exactly matches the requested byte interval and authoritative file size, and
   each body length is exact. Use parallel ranges only if aggregate throughput
   improves. For instance/cloud copy, benchmark a small non-sensitive file before
   moving assets. For laptop SFTP, measure the provided 64 MiB random file first.
4. **Pin and resume.** Pin the full Hugging Face commit SHA. Write to a separate
   staging path. Persist completed range offsets, never run two writers to one
   destination, and keep the expected file size and SHA-256 in a manifest. Do not
   log signed URLs or authorization headers.
5. **Handle authentication privately.** Use Hugging Face's protected token store
   or load the token inside the downloader from a permission-restricted secret
   source. Do not put the token in command-line arguments or logs. Preserve SSH
   authentication in the SSH agent/key store for Vast transfers; do not copy the
   private key into a dataset bundle.
6. **Verify before use.** Check every range's HTTP status, `Content-Range`, and
   byte count; verify the final archive size and authoritative SHA-256; then check
   archive readability and expected extracted dataset paths. Only after these
   checks should provisioning or an experiment consume the asset.

## Practical recommendation

Use a persistent Vast local volume when the next instance can run on the same
physical host. If it cannot, stage a verified copy in connected cloud storage or
use Vast's instance/volume copy path after measuring it. Keep pinned resumable HTTP
as the direct source fallback: it succeeded here after Xet stalled. Treat a laptop
copy as an off-cloud backup, and size its ETA from the measured SFTP test rather
than from a different network route.

