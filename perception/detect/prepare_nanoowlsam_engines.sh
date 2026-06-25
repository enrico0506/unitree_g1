#!/usr/bin/env bash
# One-time: populate perception/detect/models/data with the three TensorRT engines the
# nanoowlsam detector needs. TensorRT engines are DEVICE/TRT-specific, so they are
# produced/extracted on THIS Jetson. Run AFTER building g1-detect-nanoowlsam:latest.
#
#   data/resnet18_image_encoder.engine      <- NanoSAM image encoder  (extracted from base)
#   data/mobile_sam_mask_decoder.engine     <- NanoSAM mask decoder    (extracted from base)
#   data/owl_image_encoder_patch32.engine   <- NanoOWL OWL-ViT encoder (built on-device)
#
# Why extract the SAM engines instead of rebuilding them: the MobileSAM mask-decoder
# ONNX->engine conversion is documented to FAIL on some TensorRT versions
# (IIOneHotLayer shape-tensor error). The engines baked into dustynv/nanosam:r35.3.1 are
# already built for this exact L4T, so reusing them sidesteps that. See BUILD_NANOOWLSAM.md.
set -euo pipefail

DETECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DETECT_DIR}/models/data"
IMAGE="${IMAGE:-g1-detect-nanoowlsam:latest}"
SAM_BASE="${SAM_BASE:-dustynv/nanosam:r35.3.1}"
DOCKER="${DOCKER:-sudo docker}"

mkdir -p "${DATA}"

# Copy a file out of an image by discovering its path with `find`, then `docker cp`.
extract() {   # image  filename-glob  dest-path
  local img="$1" glob="$2" dest="$3"
  if [ -f "${dest}" ]; then echo "[prepare]  exists, skip: ${dest}"; return 0; fi
  echo "[prepare] locating ${glob} in ${img} ..."
  local src
  src="$(${DOCKER} run --rm "${img}" bash -lc "find / -name '${glob}' 2>/dev/null | head -1" | tr -d '\r')"
  if [ -z "${src}" ]; then
    echo "[prepare]  NOT FOUND: ${glob} in ${img}" >&2
    return 1
  fi
  echo "[prepare]  found ${src} -> ${dest}"
  local cid; cid="$(${DOCKER} create "${img}")"
  ${DOCKER} cp "${cid}:${src}" "${dest}"
  ${DOCKER} rm "${cid}" >/dev/null
}

echo "=== 1/3  NanoSAM ResNet18 image encoder ==="
extract "${SAM_BASE}" "resnet18_image_encoder.engine" "${DATA}/resnet18_image_encoder.engine" \
  || echo "[prepare]  -> build manually per BUILD_NANOOWLSAM.md (trtexec --fp16 on the ONNX)"

echo "=== 2/3  NanoSAM MobileSAM mask decoder ==="
extract "${SAM_BASE}" "mobile_sam_mask_decoder.engine" "${DATA}/mobile_sam_mask_decoder.engine" \
  || echo "[prepare]  -> reuse the baked engine; rebuilding the decoder may hit the TRT IIOneHotLayer bug"

echo "=== 3/3  NanoOWL OWL-ViT image encoder (built on-device with GPU) ==="
if [ -f "${DATA}/owl_image_encoder_patch32.engine" ]; then
  echo "[prepare]  exists, skip: ${DATA}/owl_image_encoder_patch32.engine"
else
  echo "[prepare] building OWL image-encoder engine inside ${IMAGE} (uses GPU, ~minutes)..."
  ${DOCKER} run --rm --runtime nvidia \
    -v "${DATA}:/out" -w /opt/nanoowl \
    "${IMAGE}" \
    bash -lc "python3 -m nanoowl.build_image_encoder_engine /out/owl_image_encoder_patch32.engine --onnx_opset=16"
fi

echo
echo "[prepare] done. Engines in ${DATA}:"
ls -lh "${DATA}" || true
echo "[prepare] If any engine is missing, see BUILD_NANOOWLSAM.md for the manual fallback."
echo "[prepare] Next: ./run_detect_nanoowlsam.sh   (verify it logs 'warmup done')"
