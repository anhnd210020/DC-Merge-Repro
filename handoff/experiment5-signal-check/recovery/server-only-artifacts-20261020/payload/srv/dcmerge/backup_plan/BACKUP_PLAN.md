# DC-Merge laptop backup plan

## Status

**DO NOT DESTROY.** This instance reports `workspace_is_volume=false`; `/workspace`
is container storage and will be lost when the instance is destroyed. Experiment 4
is safely on GitHub, but no laptop transfer or laptop hash verification has happened.
No project data, cache, or partial file was deleted.

## Payload and manifests

The selected payload is **19,197,225,149 bytes (17.879 GiB)** across 107 files,
plus metadata. Leave at least **21 GiB free on D:** for the payload, metadata,
filesystem overhead, and the 64 MiB speed-test file. The instance cannot inspect
the laptop's D: drive, so fit remains unverified until the PowerShell free-space
check is run.

- `backup_manifest.csv` lists every transferred file's source path, destination
  path, byte size, SHA-256, and category. SHA-256:
  `cb495846836508503ded6fb22ac3c3a898fc336b8561b1d42a525b9f7fd78157`.
- `SHA256SUMS` has one entry per payload file.
- `dataset_contents.csv` lists all **119,657** files in the seven non-SUN datasets
  with their original relative paths, byte sizes, and SHA-256 values. Its SHA-256
  is `44bff04cd8f0858e90037bd30965c8cd9b541f17a969a366baf5aa650b003462`.
- The non-SUN dataset tar is verified against those 119,657 entries. The SUN397
  archive is pinned to revision `42fc343338f31c2f41cfa85e97d20e22dfbf9d45`,
  size `14,591,037,440`, SHA-256
  `38afed47c5acce92ac26b36b620e936f4a849e7934e30c65487a17e2dbe0c917`.
- `excluded_inventory.csv` records preserved caches, alternate representations,
  duplicate results, and the Linux environment that are not part of the laptop
  payload. `backup_summary.json` gives category totals.

Payload contents:

| Category | Bytes | Representation |
|---|---:|---|
| Non-SUN datasets | 3,914,311,680 | One uncompressed tar; per-file checksums in `dataset_contents.csv` |
| SUN397 | 14,591,037,440 | Verified pinned archive |
| CLIP base model | 608,872,212 | Used PyTorch weights, tokenizer, and configuration |
| Adapters | 38,033,928 | Eight task adapters |
| Heads | 1,565,245 | Eight task heads |
| Git bundle | 17,999,385 | All local refs, including `main`, Experiment 3, and Experiment 4 handoff branches |
| Unique server results and logs | 19,780,067 | 40 files not already represented by Git history |
| Xet diagnostic logs | 4,356,146 | 12 log files |
| Historical Experiment 2 source archive | 1,074,449 | Readable 245-member tar.gz; local checksum recorded |
| Environment manifests | 7,399 | Linux environment exports and Python version |
| Download/recovery source and configuration | 19,662 | Five small files |
| Project helper scripts | 9,857 | Six small diagnostic/run scripts |
| Unique legacy SUN397 image | 156,160 | One changed file from the old local subset |
| Adapter metadata | 1,519 | Model repository metadata |

The archive and extracted tree for SUN397 are not both transferred. The Flax
serialization of CLIP is inventoried but omitted because the PyTorch checkpoint
used by the experiment is included. Thirty-five legacy SUN subset images are
byte-identical to the verified SUN397 tree; the one changed image is included.
The stalled 12,406,746,644-byte Xet partials, smaller incomplete model downloads,
and all other caches remain untouched on the server and are excluded from the
payload.

The installed `dcmerge_vision` environment is a 7,768,455,808-byte Linux
environment. It is not a Windows runtime, so the backup includes its small
environment exports instead. The 421 MB micromamba package cache and 53 MB HF
bootstrap environment are recreatable and excluded.

## First laptop action: capacity and speed test

Use the existing Windows OpenSSH key/agent configuration for this current
connection. The server endpoint is `root@223.166.245.253`, SSH port `40181`.
No Hugging Face token is needed for this SFTP transfer; no key or token belongs in
the commands below.

Run this first. It downloads the small manifests, checks the manifest digest,
measures one uncompressed 64 MiB SFTP file, and prints the resulting projected
ETA. It does not start the full backup.

```powershell
$ErrorActionPreference = 'Stop'
$Target = 'root@223.166.245.253'
$Port = 40181
$Backup = 'D:\Research\DC-Merge-Backup'
$Metadata = Join-Path $Backup 'metadata'
New-Item -ItemType Directory -Force $Metadata | Out-Null

$Drive = Get-PSDrive -Name D
"D: free GiB = {0:N2}" -f ($Drive.Free / 1GB)

$bootstrap = @"
lcd D:/Research/DC-Merge-Backup/metadata
get -a /srv/dcmerge/backup_plan/backup_manifest.csv
get -a /srv/dcmerge/backup_plan/backup_manifest.csv.sha256
get -a /srv/dcmerge/backup_plan/dataset_contents.csv
get -a /srv/dcmerge/backup_plan/dataset_contents.csv.sha256
get -a /srv/dcmerge/backup_plan/SHA256SUMS
get -a /srv/dcmerge/backup_plan/excluded_inventory.csv
get -a /srv/dcmerge/backup_plan/backup_summary.json
get -a /srv/dcmerge/backup_plan/BACKUP_PLAN.md
get -a /srv/dcmerge/backup_plan/next_vast_runbook.md
get -a /srv/dcmerge/backup_plan/METADATA_SHA256SUMS
bye
"@
$bootstrap | & sftp.exe -b - -P $Port -o BatchMode=yes `
  -o ServerAliveInterval=30 $Target
if ($LASTEXITCODE -ne 0) { throw "SFTP metadata download failed: $LASTEXITCODE" }

$ExpectedManifest = 'cb495846836508503ded6fb22ac3c3a898fc336b8561b1d42a525b9f7fd78157'
$ObservedManifest = (Get-FileHash -Algorithm SHA256 (Join-Path $Metadata 'backup_manifest.csv')).Hash.ToLower()
if ($ObservedManifest -ne $ExpectedManifest) { throw 'backup_manifest.csv SHA-256 mismatch' }

$testName = 'transfer-test-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.bin'
$testBatch = @"
lcd D:/Research/DC-Merge-Backup
get -a /srv/dcmerge/backup_plan/transfer_test_64MiB.bin $testName
bye
"@
$timer = [Diagnostics.Stopwatch]::StartNew()
$testBatch | & sftp.exe -b - -P $Port -o BatchMode=yes `
  -o ServerAliveInterval=30 $Target
$testExit = $LASTEXITCODE
$timer.Stop()
if ($testExit -ne 0) { throw "SFTP speed test failed: $testExit" }

$MiBPerSecond = 64 / $timer.Elapsed.TotalSeconds
$RequiredBytes = [Int64]19197225149
$EtaHours = ($RequiredBytes / 1MB) / $MiBPerSecond / 3600
[pscustomobject]@{
  TestSeconds = [math]::Round($timer.Elapsed.TotalSeconds, 2)
  MeasuredMiBPerSecond = [math]::Round($MiBPerSecond, 2)
  EstimatedPayloadHours = [math]::Round($EtaHours, 2)
  RequiredPayloadGiB = [math]::Round($RequiredBytes / 1GB, 2)
  DFreeGiB = [math]::Round((Get-PSDrive D).Free / 1GB, 2)
}
```

Send back the displayed `MeasuredMiBPerSecond`, `EstimatedPayloadHours`, and
`DFreeGiB`. The previous laptop-to-server transfer was described as very slow,
but no numeric rate was recorded. I will not substitute the unrelated Hugging Face
rate for this laptop measurement.

## Full transfer, after capacity and ETA are confirmed

The manifest transfer can be resumed by rerunning the same block. `sftp get -a`
continues a partial local file; it assumes that the existing partial bytes match
the unchanged server file. Keep one SFTP transfer writer active and do not modify
destination files while resuming. The official [OpenSSH sftp manual](https://man.openbsd.org/OpenBSD-7.5/sftp.1)
documents this resume behavior.

```powershell
$ErrorActionPreference = 'Stop'
$Target = 'root@223.166.245.253'
$Port = 40181
$Backup = 'D:\Research\DC-Merge-Backup'
$Metadata = Join-Path $Backup 'metadata'
$Manifest = Import-Csv (Join-Path $Metadata 'backup_manifest.csv')
$RequiredBytes = [Int64](($Manifest | Measure-Object -Property bytes -Sum).Sum)
if ($RequiredBytes -ne 19197225149) { throw "Unexpected manifest payload size: $RequiredBytes" }
$NeedWithHeadroom = [Int64][math]::Ceiling($RequiredBytes * 1.10)
if ((Get-PSDrive D).Free -lt $NeedWithHeadroom) {
  throw "D: needs at least $([math]::Round($NeedWithHeadroom / 1GB, 2)) GiB free before transfer"
}

$Batch = Join-Path $Metadata 'full_transfer.sftp'
$Lines = [System.Collections.Generic.List[string]]::new()
$Lines.Add('progress')  # show per-file transfer progress in the SFTP log
foreach ($Row in $Manifest) {
  $LocalPath = Join-Path $Backup ($Row.backup_path -replace '/', '\')
  $LocalParent = Split-Path -Parent $LocalPath
  New-Item -ItemType Directory -Force $LocalParent | Out-Null
  $LocalParentSftp = $LocalParent -replace '\', '/'
  $Lines.Add("lcd $LocalParentSftp")
  $Lines.Add("get -a `"$($Row.source_path)`"")
}
[IO.File]::WriteAllLines($Batch, $Lines, [Text.Encoding]::ASCII)
& sftp.exe -b $Batch -P $Port -o BatchMode=yes `
  -o ServerAliveInterval=30 $Target 2>&1 |
  Tee-Object -FilePath (Join-Path $Backup 'transfer.log')
if ($LASTEXITCODE -ne 0) { throw "SFTP backup transfer failed: $LASTEXITCODE; rerun this block to resume" }
```

## Local verification

Run after the SFTP block finishes. It verifies every transferred file's exact
length and SHA-256 against `backup_manifest.csv`.

```powershell
$Backup = 'D:\Research\DC-Merge-Backup'
$Manifest = Import-Csv (Join-Path $Backup 'metadata\backup_manifest.csv')
$Failures = [System.Collections.Generic.List[string]]::new()
foreach ($Row in $Manifest) {
  $Path = Join-Path $Backup ($Row.backup_path -replace '/', '\')
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    $Failures.Add("MISSING $($Row.backup_path)")
    continue
  }
  if ((Get-Item -LiteralPath $Path).Length -ne [Int64]$Row.bytes) {
    $Failures.Add("SIZE $($Row.backup_path)")
    continue
  }
  $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLower()
  if ($Hash -ne $Row.sha256) { $Failures.Add("SHA256 $($Row.backup_path)") }
}
if ($Failures.Count) { $Failures; throw "$($Failures.Count) backup verification failures" }
"PASS: verified $($Manifest.Count) payload files; bytes=$((($Manifest | Measure-Object -Property bytes -Sum).Sum))"

# Optional after extracting datasets8_non_sun397.tar:
# Compare each extracted file with metadata\dataset_contents.csv using
# Get-FileHash; the archive's own SHA-256 is already checked above.
```

The backup is complete only after the laptop prints `PASS` and you provide that
verification output. Until then the status remains **DO NOT DESTROY**.

