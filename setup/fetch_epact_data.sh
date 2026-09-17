#!/usr/bin/env bash
# EPACT publishes its checkpoints and training data on Zenodo, not in its repository.
# Record 10996150 (EPACT v0.1.0, CC-BY-4.0), as named in the EPACT README.
#
#   ./setup/fetch_epact_data.sh              checkpoints (2.4 GB) + data (171 MB)
#   ./setup/fetch_epact_data.sh --data-only  just the 171 MB data archive
#
# Each archive is checked against the md5 Zenodo publishes before it is unpacked.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPACT_DIR="${TCRJ_MODELS:-$ROOT/models}/EPACT"
RECORD=10996150
BASE="https://zenodo.org/records/$RECORD/files"

# file : md5 published by Zenodo : directory to unpack into
# The archives are rooted at their own contents, and the runner reads checkpoints/pretrained/*.pt
# and data/hla_library.json - so each one is unpacked into the directory it belongs in.
CHECKPOINTS="EPACT-model-checkpoints.zip:dec456af5b34bb2787b99c795e24e5c9:checkpoints"
DATA="EPACT-data.zip:db1a5f25f128712a809e48c3cd0192bf:data"

WANTED=("$CHECKPOINTS" "$DATA")
[ "${1:-}" = "--data-only" ] && WANTED=("$DATA")

[ -d "$EPACT_DIR" ] || { echo "EPACT not fetched yet: ./setup/fetch_models.sh EPACT"; exit 1; }
command -v curl >/dev/null || { echo "curl is required"; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required"; exit 1; }
cd "$EPACT_DIR"

for entry in "${WANTED[@]}"; do
    IFS=':' read -r file want_md5 dest <<< "$entry"
    if [ ! -f "$file" ]; then
        echo "downloading $file from Zenodo record $RECORD"
        curl -fSL --retry 3 -o "$file.part" "$BASE/$file"
        mv "$file.part" "$file"
    else
        echo "$file already downloaded"
    fi
    have_md5="$(md5sum "$file" | cut -d' ' -f1)"
    if [ "$have_md5" != "$want_md5" ]; then
        echo "CHECKSUM MISMATCH for $file"
        echo "  expected $want_md5"
        echo "  got      $have_md5"
        echo "Delete the file and retry; do not use it."
        exit 1
    fi
    echo "  md5 ok ($have_md5)"
    echo "  unpacking into $dest/"
    mkdir -p "$dest"
    unzip -q -o "$file" -d "$dest" -x '__MACOSX/*'
done

echo
echo "EPACT assets under $EPACT_DIR:"
ls -d checkpoints data 2>/dev/null || true
echo
echo "Check what the pipeline still needs:  python3 setup/validate_models.py"
