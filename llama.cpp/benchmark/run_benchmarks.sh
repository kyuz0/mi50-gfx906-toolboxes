#!/usr/bin/env bash
set -uo pipefail

MODEL_DIR="$(realpath ~/models)"
RESULTDIR="results"
mkdir -p "$RESULTDIR"

# Pick exactly one .gguf per model: either
#  - any .gguf without "-000*-of-" (single-file models)
#  - or the first shard "*-00001-of-*.gguf"
mapfile -t MODEL_PATHS < <(
  find "$MODEL_DIR" -type f -name '*.gguf' \
    \( -name '*-00001-of-*.gguf' -o -not -name '*-000*-of-*.gguf' \) \
    | sort
)

if (( ${#MODEL_PATHS[@]} == 0 )); then
  echo "❌ No models found under $MODEL_DIR – check your paths/patterns!"
  exit 1
fi

echo "Found ${#MODEL_PATHS[@]} model(s) to bench:"
for p in "${MODEL_PATHS[@]}"; do
  echo "  • $p"
done
echo

declare -A CMDS=(
  [rocm6_4_4]="toolbox run -c llama-rocm-6.4.4 -- /usr/local/bin/llama-bench"
  [rocm7_2_1]="toolbox run -c llama-rocm-7.2 -- /usr/local/bin/llama-bench"
)



for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH" .gguf)"

  if [[ "$MODEL_PATH" == *"-00001-of-"* ]]; then
    # Multi-shard model: sum all shards
    DIR="$(dirname "$MODEL_PATH")"
    BASE="$(basename "$MODEL_PATH")"
    PATTERN="${BASE/-00001-of-/-*-of-}"
    # Pure bash addition to avoid `awk` scientific notation formatting bugs
    MODEL_SIZE=0
    for file in "$DIR"/$PATTERN; do
      if [[ -f "$file" ]]; then
        size=$(stat -c%s "$file")
        ((MODEL_SIZE+=size))
      fi
    done
  else
    # Single-file model
    MODEL_SIZE=$(stat -c%s "$MODEL_PATH")
  fi

  # Threshold increments of ~14 GiB (approx 15000000000 bytes).
  if (( MODEL_SIZE > 45000000000 )); then
    GPU_DEVICES="0,1,2,3"
    GPU_SUFFIX="__quad"
  elif (( MODEL_SIZE > 30000000000 )); then
    GPU_DEVICES="0,1,2"
    GPU_SUFFIX="__triple"
  elif (( MODEL_SIZE > 15000000000 )); then
    GPU_DEVICES="0,1"
    GPU_SUFFIX="__dual"
  else
    GPU_DEVICES="0"
    GPU_SUFFIX="__single"
  fi

  for ENV in "${!CMDS[@]}"; do
    CMD="${CMDS[$ENV]}"
    CMD_EFFECTIVE="${CMD/-- /-- env HIP_VISIBLE_DEVICES=$GPU_DEVICES }"

    # run with flash attention
    for FA in 1; do
      SUFFIX="__fa1"
      EXTRA_ARGS=( -fa 1 )

      for CTX in default longctx32768; do
        CTX_SUFFIX=""
        CTX_ARGS=()
        if [[ "$CTX" == longctx32768 ]]; then
          CTX_SUFFIX="__longctx32768"
          CTX_ARGS=( -p 2048 -n 32 -d 32768 )
          if [[ "$ENV" == *vulkan* ]]; then
            CTX_ARGS+=( -ub 512 )
          else
            CTX_ARGS+=( -ub 2048 )
          fi
        fi

        OUT="$RESULTDIR/${MODEL_NAME}__${ENV}${SUFFIX}${CTX_SUFFIX}${GPU_SUFFIX}.log"
        CTX_REPS=3
        if [[ "$CTX" == longctx* ]]; then
          CTX_REPS=1
        fi

        if [[ -s "$OUT" ]]; then
          echo "⏩ Skipping [${ENV}] ${MODEL_NAME}${SUFFIX}${CTX_SUFFIX:+ ($CTX_SUFFIX)}, log already exists at $OUT"
          continue
        fi

        FULL_CMD=( $CMD_EFFECTIVE -ngl 99 -m "$MODEL_PATH" "${EXTRA_ARGS[@]}" "${CTX_ARGS[@]}" -r "$CTX_REPS" )

        printf "\n▶ [%s] %s%s%s\n" "$ENV" "$MODEL_NAME" "${SUFFIX:+ $SUFFIX}" "${CTX_SUFFIX:+ $CTX_SUFFIX}"
        printf "  → log: %s\n" "$OUT"
        printf "  → cmd: %s\n\n" "${FULL_CMD[*]}"

        if ! "${FULL_CMD[@]}" >"$OUT" 2>&1; then
          status=$?
          echo "✖ ! [${ENV}] ${MODEL_NAME}${SUFFIX}${CTX_SUFFIX:+ $CTX_SUFFIX} failed (exit ${status})" >>"$OUT"
          echo "  * [${ENV}] ${MODEL_NAME}${SUFFIX}${CTX_SUFFIX:+ $CTX_SUFFIX} : FAILED"
        fi
      done
    done
  done
done
