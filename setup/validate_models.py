#!/usr/bin/env python3
"""What is present under models/, and what stage 3 still needs.

Stage 3 (model inference) is the one step this repository cannot run on its own: the eight
predictors live in their own public repositories. This reports, per model, whether the repository is
there, which commit is checked out, and whether the specific files the runners and the
training-overlap analysis read are present.

  ./setup/fetch_models.sh            # clone them
  python3 setup/validate_models.py   # then this

Nothing here is needed to rebuild the manuscript: data/scores/ already holds the model outputs.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import os  # noqa: E402

MODELS_DIR = Path(os.environ.get("TCRJ_MODELS", REPO / "models"))
SOURCES = REPO / "setup" / "model_sources.tsv"

# Files each model needs, relative to its own directory.
#   training: read by stage 2 --recompute and by analysis/overlap.py
#   weights:  read by the runner at inference time
EXPECTED = {
    "ERGO":     {"training": ["data/VDJDB_complete.tsv"], "weights": []},
    "ERGO2":    {"training": ["mcpas_train.csv", "Samples/mcpas_train_samples.pickle"], "weights": []},
    "NetTCR":   {"training": ["data/train_ab_95_alphabeta.csv"], "weights": []},
    "NetTCR22": {"training": ["data/nettcr_2_2_full_dataset.csv"], "weights": []},
    "TITAN":    {"training": ["datasets/full_data+covid.csv"], "weights": []},
    "EPACT":    {"training": ["sample/VDJdb-GLCTLVAML.csv"],
                 "weights": ["checkpoints/pretrained/pmhc-BA-model-medium.pt",
                             "checkpoints/pretrained/paired-cdr3-model-medium.pt",
                             "data/hla_library.json"]},
    "PanPep":   {"training": ["Data/majority_training_dataset.csv"], "weights": []},
    "SCEPTR":   {"training": [], "weights": []},   # pip package, embeddings bundled
}


def sources():
    rows = []
    for line in SOURCES.read_text().splitlines():
        if not line or line.startswith("#") or line.startswith("key\t"):
            continue
        key, url, commit, *rest = line.split("\t")
        rows.append((key, url, commit, rest[0] if rest else ""))
    return rows


def main():
    print(f"models directory: {MODELS_DIR}"
          f"{'' if MODELS_DIR.exists() else '   (does not exist yet)'}\n")
    ready = missing = 0
    for key, url, commit, notes in sources():
        d = MODELS_DIR / key
        if url == "pip:sceptr":
            print(f"{key:9s} pip package - install with `pip install sceptr` in its environment")
            continue
        if not d.exists():
            print(f"{key:9s} NOT FETCHED   ./setup/fetch_models.sh {key}")
            missing += 1
            continue
        head = "?"
        if (d / ".git").exists():
            r = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True)
            head = r.stdout.strip() or "?"
            if commit != "-" and not commit.startswith(head):
                head += f" (pin says {commit[:7]})"
        exp = EXPECTED.get(key, {"training": [], "weights": []})
        absent = [f for f in exp["training"] + exp["weights"] if not (d / f).exists()]
        if absent:
            print(f"{key:9s} present at {head}, but {len(absent)} expected file(s) missing:")
            for f in absent:
                print(f"{'':11s}  {f}")
            if key == "EPACT":
                print(f"{'':11s}  -> ./setup/fetch_epact_data.sh")
            missing += 1
        else:
            print(f"{key:9s} ready at {head}")
            ready += 1
    print(f"\n{ready} model(s) ready, {missing} incomplete")
    if missing:
        print("Stage 3 needs every model it is asked to run; the rest of the pipeline does not "
              "need any of them (data/scores/ is committed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
