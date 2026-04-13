#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR}"

URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M10/gencode.vM10.chr_patch_hapl_scaff.annotation.gtf.gz"
GZ_NAME="gencode.vM10.chr_patch_hapl_scaff.annotation.gtf.gz"
GTF_NAME="gencode.vM10.chr_patch_hapl_scaff.annotation.gtf"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

download_with_wget() {
    wget -c -O "$GZ_NAME" "$URL"
}

download_with_curl() {
    curl -L -C - -o "$GZ_NAME" "$URL"
}

echo "Output directory: $OUT_DIR"
echo "Downloading: $URL"

if command -v wget >/dev/null 2>&1; then
    download_with_wget
elif command -v curl >/dev/null 2>&1; then
    download_with_curl
else
    echo "ERROR: neither wget nor curl is available." >&2
    exit 1
fi

echo "Decompressing to: $OUT_DIR/$GTF_NAME"
gzip -dc "$GZ_NAME" > "$GTF_NAME"

echo "Done."
echo "GZ : $OUT_DIR/$GZ_NAME"
echo "GTF: $OUT_DIR/$GTF_NAME"
