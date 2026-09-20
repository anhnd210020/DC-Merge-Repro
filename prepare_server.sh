#!/usr/bin/env bash
set -euo pipefail

# DC-Merge server bootstrap for:
#   ViT-B/32 + LoRA r=16 + 8 vision tasks
#
# Before running on a fresh server with private Hugging Face repos:
#   export HF_TOKEN='hf_...'
#   bash prepare_server.sh
#
# The token is never written into this script or into the repo.

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/DC-Merge-Repro}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/datasets8}"
MODELS_DIR="${MODELS_DIR:-$WORKSPACE/models}"
CLIP_DIR="${CLIP_DIR:-$MODELS_DIR/clip-vit-base-patch32}"
MODEL_BUNDLE_DIR="${MODEL_BUNDLE_DIR:-$MODELS_DIR/DC-Merge-Vision-B32-r16-8task}"
TMP_TAR_DIR="${TMP_TAR_DIR:-$WORKSPACE/.dcmerge_dataset_tars}"
BOOTSTRAP_VENV="${BOOTSTRAP_VENV:-$WORKSPACE/.hf_bootstrap}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$WORKSPACE/.micromamba}"
MICROMAMBA="${MICROMAMBA:-$WORKSPACE/bin/micromamba}"

GIT_REPO="${GIT_REPO:-https://github.com/anhnd210020/DC-Merge-Repro.git}"
DATA_REPO="${DATA_REPO:-anhnd210020/DC-Merge-Vision-8Task}"
MODEL_REPO="${MODEL_REPO:-anhnd210020/DC-Merge-Vision-B32-r16-8task}"
CLIP_REPO="${CLIP_REPO:-openai/clip-vit-base-patch32}"

TASKS=(stanford_cars dtd eurosat gtsrb mnist resisc45 sun397 svhn)
DATA_TARS=(cars.tar dtd.tar eurosat.tar gtsrb.tar MNIST.tar resisc45.tar sun397.tar svhn.tar)

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

log "Preflight"
mkdir -p "$WORKSPACE" "$DATA_DIR" "$MODELS_DIR" "$TMP_TAR_DIR" "$WORKSPACE/bin"
command -v git >/dev/null || { echo "ERROR: git not found"; exit 1; }
command -v curl >/dev/null || { echo "ERROR: curl not found"; exit 1; }
command -v tar >/dev/null || { echo "ERROR: tar not found"; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 1; }

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found yet."
fi
df -h "$WORKSPACE" | tail -n 1 || true

log "Clone/update DC-Merge repro"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch origin main
  git -C "$REPO_DIR" checkout main
  git -C "$REPO_DIR" pull --ff-only origin main
else
  git clone "$GIT_REPO" "$REPO_DIR"
fi
echo "Repo commit: $(git -C "$REPO_DIR" rev-parse --short HEAD)"

log "Create isolated Hugging Face bootstrap environment"
if [[ ! -x "$BOOTSTRAP_VENV/bin/python" ]]; then
  if ! python3 -m venv "$BOOTSTRAP_VENV"; then
    if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
      apt-get update
      apt-get install -y python3-venv
      python3 -m venv "$BOOTSTRAP_VENV"
    else
      echo "ERROR: python3-venv is unavailable."
      exit 1
    fi
  fi
fi
"$BOOTSTRAP_VENV/bin/python" -m pip install -q -U pip
"$BOOTSTRAP_VENV/bin/python" -m pip install -q -U "huggingface_hub==1.32.0" "hf_xet==1.6.0"
HF="$BOOTSTRAP_VENV/bin/hf"

if ! "$HF" auth whoami >/dev/null 2>&1; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo
    echo "ERROR: private Hugging Face repos require authentication."
    echo "Run one of these, then rerun this script:"
    echo "  export HF_TOKEN='hf_...'"
    echo "or"
    echo "  $HF auth login"
    exit 2
  fi
fi

log "Download + extract the 8 datasets one-by-one"
for tar_name in "${DATA_TARS[@]}"; do
  folder="${tar_name%.tar}"
  if [[ -d "$DATA_DIR/$folder" ]]; then
    echo "SKIP: $folder already exists"
    continue
  fi

  echo ">>> $tar_name"
  "$HF" download "$DATA_REPO" "$tar_name" \
    --repo-type dataset \
    --local-dir "$TMP_TAR_DIR"

  test -s "$TMP_TAR_DIR/$tar_name"
  tar -xf "$TMP_TAR_DIR/$tar_name" -C "$DATA_DIR"
  rm -f "$TMP_TAR_DIR/$tar_name"
done

log "Download LoRA checkpoints + classification heads"
mkdir -p "$MODEL_BUNDLE_DIR"
"$HF" download "$MODEL_REPO" \
  --repo-type model \
  --local-dir "$MODEL_BUNDLE_DIR"

log "Download CLIP ViT-B/32"
mkdir -p "$CLIP_DIR"
"$HF" download "$CLIP_REPO" \
  --repo-type model \
  --local-dir "$CLIP_DIR"

log "Install micromamba if needed"
if [[ ! -x "$MICROMAMBA" ]]; then
  tmpdir="$(mktemp -d)"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "$tmpdir" bin/micromamba
  mv "$tmpdir/bin/micromamba" "$MICROMAMBA"
  chmod +x "$MICROMAMBA"
  rm -rf "$tmpdir"
fi

export MAMBA_ROOT_PREFIX
ENV_YML="$REPO_DIR/environment_vision.yml"
ENV_NAME="$(sed -n 's/^name:[[:space:]]*//p' "$ENV_YML" | head -n 1 | tr -d "\"'")"
if [[ -z "$ENV_NAME" ]]; then
  echo "ERROR: could not read environment name from $ENV_YML"
  exit 1
fi
echo "Vision environment: $ENV_NAME"

log "Create the official vision environment"
if "$MICROMAMBA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "SKIP: micromamba env '$ENV_NAME' already exists"
else
  "$MICROMAMBA" create -y -f "$ENV_YML"
fi

log "Write reusable path environment"
ENV_FILE="$WORKSPACE/dcmerge_env.sh"
cat > "$ENV_FILE" <<EOF
export DCMERGE_REPO="$REPO_DIR"
export DCMERGE_DATA_DIR="$DATA_DIR"
export DCMERGE_MODEL_DIR="$CLIP_DIR"
export DCMERGE_CACHE_DIR="$CLIP_DIR"
export DCMERGE_HEAD_DIR="$MODEL_BUNDLE_DIR/lora_heads"
export DCMERGE_FT_DIR="$MODEL_BUNDLE_DIR/lora_checkpoints"
export DCMERGE_ENV_NAME="$ENV_NAME"
export DCMERGE_MICROMAMBA="$MICROMAMBA"
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX"
EOF

source "$ENV_FILE"

log "Validate downloaded assets"
"$MICROMAMBA" run -n "$ENV_NAME" python - <<'PY'
import os
from pathlib import Path

tasks = [
    "stanford_cars", "dtd", "eurosat", "gtsrb",
    "mnist", "resisc45", "sun397", "svhn",
]

data = Path(os.environ["DCMERGE_DATA_DIR"])
ft = Path(os.environ["DCMERGE_FT_DIR"])
heads = Path(os.environ["DCMERGE_HEAD_DIR"]) / "ViT-B-32"
repo = Path(os.environ["DCMERGE_REPO"])

dataset_folders = {
    "stanford_cars": data / "cars",
    "dtd": data / "dtd",
    "eurosat": data / "eurosat",
    "gtsrb": data / "gtsrb",
    "mnist": data / "MNIST",
    "resisc45": data / "resisc45",
    "sun397": data / "sun397",
    "svhn": data / "svhn",
}

errors = []

print("=== DATASETS ===")
for task, path in dataset_folders.items():
    ok = path.is_dir()
    print(f"{task:15} {'PASS' if ok else 'MISSING'}  {path}")
    if not ok:
        errors.append(str(path))

print("\n=== LORA CHECKPOINTS ===")
for task in tasks:
    cfg = ft / task / "adapter_config.json"
    model = ft / task / "adapter_model.bin"
    ok = cfg.is_file() and model.is_file()
    print(f"{task:15} {'PASS' if ok else 'MISSING'}")
    if not ok:
        errors.extend([str(cfg), str(model)])

print("\n=== HEADS ===")
for task in tasks:
    path = heads / f"{task}_head.pt"
    ok = path.is_file()
    print(f"{task:15} {'PASS' if ok else 'MISSING'}")
    if not ok:
        errors.append(str(path))

for name in ("val_acc.json", "test_acc.json"):
    path = repo / "vision_lora_merge" / "single_task" / "our_finetuned" / "ViT-B-32" / name
    ok = path.is_file()
    print(f"\n{name:15} {'PASS' if ok else 'MISSING'}")
    if not ok:
        errors.append(str(path))

if errors:
    raise SystemExit("ASSET VALIDATION FAILED:\n" + "\n".join(errors))

print("\nASSET VALIDATION PASS")
PY

log "Validate local CLIP and project dependencies"
"$MICROMAMBA" run -n "$ENV_NAME" python - <<'PY'
import os
import torch
import transformers
import peft
from transformers import CLIPModel, CLIPProcessor

p = os.environ["DCMERGE_CACHE_DIR"]
model = CLIPModel.from_pretrained(p, local_files_only=True)
_ = CLIPProcessor.from_pretrained(p, local_files_only=True)

print("torch =", torch.__version__)
print("transformers =", transformers.__version__)
print("peft =", peft.__version__)
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))

print("CLIP LOCAL LOAD PASS")
print("hidden_size =", model.config.vision_config.hidden_size)
print("image_size =", model.config.vision_config.image_size)
print("patch_size =", model.config.vision_config.patch_size)
PY

log "Validate DC-Merge LoRA wrapper"
(
  cd "$REPO_DIR/vision_lora_merge"
  "$MICROMAMBA" run -n "$ENV_NAME" python - <<'PY'
from configs import vitB32_r16_8task as c
from models.huggingface_clip import get_model_from_config

m = get_model_from_config(c.config["model"], "cpu")
print("DC-MERGE LORA MODEL INIT PASS")
print(type(m).__name__)
PY
)

log "Create one-command evaluation launcher"
RUNNER="$WORKSPACE/run_dcmerge_b32_8task.sh"
cat > "$RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /workspace/dcmerge_env.sh
cd "$DCMERGE_REPO/vision_lora_merge"
exec "$DCMERGE_MICROMAMBA" run -n "$DCMERGE_ENV_NAME" \
  python eval.py \
  --config vitB32_r16_8task \
  --method dc_merge \
  --smoothing linear \
  "$@"
EOF
chmod +x "$RUNNER"

log "Final status"
echo "Repo:       $REPO_DIR"
echo "Datasets:   $DATA_DIR"
echo "CLIP:       $CLIP_DIR"
echo "LoRA:       $DCMERGE_FT_DIR"
echo "Heads:      $DCMERGE_HEAD_DIR"
echo "Env:        $ENV_NAME"
echo "Env exports:$ENV_FILE"
echo "Runner:     $RUNNER"
echo
df -h "$WORKSPACE" | tail -n 1 || true
echo
echo "SETUP COMPLETE"
echo "When ready to evaluate:"
echo "  bash $RUNNER"
