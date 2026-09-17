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

# file : md5 published by Zenodo
# Both archives unpack at the EPACT root, which is where the runner looks: it passes
# --model_location paired-cdr3-pmhc-binding/...-fold-1.pt. The config it generates additionally
# refers to checkpoints/pretrained/... and data/..., so those are linked to the same files
# afterwards - this reproduces the layout of the clone that produced data/scores/.
CHECKPOINTS="EPACT-model-checkpoints.zip:dec456af5b34bb2787b99c795e24e5c9"
DATA="EPACT-data.zip:db1a5f25f128712a809e48c3cd0192bf"

WANTED=("$CHECKPOINTS" "$DATA")
[ "${1:-}" = "--data-only" ] && WANTED=("$DATA")

[ -d "$EPACT_DIR" ] || { echo "EPACT not fetched yet: ./setup/fetch_models.sh EPACT"; exit 1; }
command -v curl >/dev/null || { echo "curl is required"; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required"; exit 1; }
cd "$EPACT_DIR"

for entry in "${WANTED[@]}"; do
    IFS=':' read -r file want_md5 <<< "$entry"
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
    echo "  unpacking at the EPACT root"
    unzip -q -o "$file" -x '__MACOSX/*'
done
rm -rf __MACOSX

# The generated prediction config uses checkpoints/ and data/ prefixes for the same files.
mkdir -p checkpoints data
for d in pretrained paired-cdr3-pmhc-binding paired-cdr123-pmhc-binding paired-cdr123-pmhc-interaction; do
    [ -d "$d" ] && [ ! -e "checkpoints/$d" ] && ln -s "../$d" "checkpoints/$d"
done
for d in binding structure; do
    [ -d "$d" ] && [ ! -e "data/$d" ] && ln -s "../$d" "data/$d"
done
[ -f hla_library.json ] && [ ! -e data/hla_library.json ] && ln -s ../hla_library.json data/hla_library.json
echo "  linked checkpoints/ and data/ to the same files"

echo
echo "EPACT assets under $EPACT_DIR:"
ls -d checkpoints data 2>/dev/null || true
echo
echo "Check what the pipeline still needs:  python3 setup/validate_models.py"
