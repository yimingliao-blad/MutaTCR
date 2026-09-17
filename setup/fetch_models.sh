#!/usr/bin/env bash
# Fetch the eight predictors' public repositories into models/ (stage 3 only).
#
#   ./setup/fetch_models.sh              every model in setup/model_sources.tsv
#   ./setup/fetch_models.sh NetTCR       just one (repeatable)
#   ./setup/fetch_models.sh --list       show the sources and exit
#
# Clones at the pinned commit, and records what was actually checked out in models/FETCHED.tsv.
# A failed clone stops the script: a half-populated models directory would make stage 3 fail in a
# much more confusing way later.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCES="$ROOT/setup/model_sources.tsv"
MODELS_DIR="${TCRJ_MODELS:-$ROOT/models}"
[ -f "$SOURCES" ] || { echo "missing $SOURCES"; exit 1; }

if [ "${1:-}" = "--list" ]; then
    printf '%-10s %-55s %s\n' KEY URL COMMIT
    grep -v '^#' "$SOURCES" | grep -v '^key' | grep -v '^$' | while IFS=$'\t' read -r k u c n; do
        printf '%-10s %-55s %s\n' "$k" "$u" "$c"
    done
    exit 0
fi

command -v git >/dev/null || { echo "git is required"; exit 1; }
WANTED=("$@")
mkdir -p "$MODELS_DIR"
MANIFEST="$MODELS_DIR/FETCHED.tsv"
[ -f "$MANIFEST" ] || printf 'key\turl\tpinned_commit\tcheckout_commit\tfetched_at\n' > "$MANIFEST"

want() {
    [ ${#WANTED[@]} -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

grep -v '^#' "$SOURCES" | grep -v '^key' | grep -v '^$' | while IFS=$'\t' read -r key url commit notes; do
    want "$key" || continue
    if [ "$url" = "pip:sceptr" ]; then
        echo "== $key: installed with pip inside its environment, nothing to clone"
        continue
    fi
    target="$MODELS_DIR/$key"
    if [ -d "$target/.git" ]; then
        echo "== $key: already present at $target"
    else
        echo "== $key: cloning $url"
        git clone --quiet "$url" "$target"
    fi
    if [ "$commit" != "-" ]; then
        git -C "$target" fetch --quiet origin "$commit" 2>/dev/null || true
        git -C "$target" checkout --quiet "$commit"
    fi
    have="$(git -C "$target" rev-parse HEAD)"
    printf '%s\t%s\t%s\t%s\t%s\n' "$key" "$url" "$commit" "$have" "$(date -Is)" >> "$MANIFEST"
    echo "   at $have"
done

echo
echo "Recorded in $MANIFEST"
echo "Next: ./setup/fetch_epact_data.sh   (EPACT checkpoints)"
echo "      python3 setup/validate_models.py   (what is ready, what is missing)"
