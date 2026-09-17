import sqlite3
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# Configuration
# ============================================================

# --- made path-configurable for the staged pipeline (originally DB_PATH = "./fp.db") ---
import os as _os
from pathlib import Path as _Path
DB_PATH = _os.environ.get("TCRJ_DB", "./fp.db")
_OUT = _Path(_os.environ.get("TCRJ_OUT", "."))
_OUT.mkdir(parents=True, exist_ok=True)
_os.chdir(_OUT)   # the script writes its CSVs relative to the working directory
DB_PATH = str(_Path(DB_PATH).resolve()) if _Path(DB_PATH).exists() else DB_PATH
REFERENCE_PEPTIDE = "YLQPRTFLL"

MODEL_TABLES = [
    "EPACT",
    "ERGO2",
    "NetTCR22",
    "SCEPTR",
    "ERGO",
    "NetTCR",
    "PanPep",
    "TITAN",
]


# ============================================================
# Mutation parsing
# ============================================================

def get_mutation_info(peptide, reference=REFERENCE_PEPTIDE):
    """
    Infer mutation information from peptide sequence.

    Returns:
        mutation        e.g. R5K
        position        e.g. 5
        wt_residue      e.g. R
        mutant_residue  e.g. K

    Returns None values if peptide is not exactly one
    amino-acid substitution from the reference.
    """

    if not isinstance(peptide, str):
        return None, None, None, None

    peptide = peptide.strip()

    if len(peptide) != len(reference):
        return None, None, None, None

    diff = [
        i for i, (wt, mut) in enumerate(zip(reference, peptide))
        if wt != mut
    ]

    if len(diff) != 1:
        return None, None, None, None

    i = diff[0]

    wt = reference[i]
    mut = peptide[i]
    pos = i + 1

    mutation = f"{wt}{pos}{mut}"

    return mutation, pos, wt, mut


def prepare_data(df):
    """
    Clean records and retain only valid single-AA mutants.
    """

    df = df.copy()

    df = df.dropna(
        subset=["Peptide", "Label", "Prediction_Prob"]
    )

    df["Peptide"] = df["Peptide"].astype(str).str.strip()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df["Prediction_Prob"] = pd.to_numeric(
        df["Prediction_Prob"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Label", "Prediction_Prob"]
    )

    df["Label"] = df["Label"].astype(int)

    mutation_info = df["Peptide"].apply(get_mutation_info)

    df["Mutation"] = mutation_info.apply(lambda x: x[0])
    df["Mutation_Position"] = mutation_info.apply(lambda x: x[1])
    df["WT_Residue"] = mutation_info.apply(lambda x: x[2])
    df["Mutant_Residue"] = mutation_info.apply(lambda x: x[3])

    # Keep only valid single-AA mutants.
    # This automatically excludes reference YLQPRTFLL.
    df = df[df["Mutation"].notna()].copy()

    return df


# ============================================================
# Per-mutation metrics
# ============================================================

def calculate_per_mutation_metrics(df):
    """
    For each mutant peptide, calculate AUC0.1 and AUPRC
    across the available TCRs.

    AUC0.1 requires both positive and negative labels.
    AUPRC is also restricted here to mutations with both
    classes so that macro metrics use the same eligible set.
    """

    rows = []

    for peptide, group in df.groupby("Peptide"):

        y_true = group["Label"].to_numpy()
        y_score = group["Prediction_Prob"].to_numpy()

        n = len(group)
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))

        mutation = group["Mutation"].iloc[0]
        position = int(group["Mutation_Position"].iloc[0])
        wt = group["WT_Residue"].iloc[0]
        mut = group["Mutant_Residue"].iloc[0]

        # Cannot calculate ROC AUC with only one class
        if len(np.unique(y_true)) < 2:
            auc01 = np.nan
            auprc = np.nan
            eligible = False
        else:
            auc01 = roc_auc_score(
                y_true,
                y_score,
                max_fpr=0.1
            )

            auprc = average_precision_score(
                y_true,
                y_score
            )

            eligible = True

        rows.append({
            "Peptide": peptide,
            "Mutation": mutation,
            "Mutation_Position": position,
            "WT_Residue": wt,
            "Mutant_Residue": mut,
            "N_TCR": n,
            "N_Positive": n_pos,
            "N_Negative": n_neg,
            "Positive_Fraction": n_pos / n if n > 0 else np.nan,
            "Eligible_for_AUC": eligible,
            "AUC0.1": auc01,
            "AUPRC": auprc,
        })

    return pd.DataFrame(rows)


# ============================================================
# Overall model metrics
# ============================================================

def calculate_overall_metrics(df, per_mutation_df):
    """
    Compute:
      1. Macro AUC0.1
      2. Macro AUPRC
      3. Pooled AUC0.1
      4. Pooled AUPRC
    """

    eligible = per_mutation_df[
        per_mutation_df["Eligible_for_AUC"]
    ].copy()

    macro_auc01 = (
        eligible["AUC0.1"].mean()
        if len(eligible) > 0 else np.nan
    )

    macro_auprc = (
        eligible["AUPRC"].mean()
        if len(eligible) > 0 else np.nan
    )

    y_true = df["Label"].to_numpy()
    y_score = df["Prediction_Prob"].to_numpy()

    if len(np.unique(y_true)) >= 2:

        pooled_auc01 = roc_auc_score(
            y_true,
            y_score,
            max_fpr=0.1
        )

        pooled_auprc = average_precision_score(
            y_true,
            y_score
        )

    else:
        pooled_auc01 = np.nan
        pooled_auprc = np.nan

    return {
        "N_Pairs": len(df),
        "N_Mutations": df["Peptide"].nunique(),
        "N_Eligible_Mutations": len(eligible),

        "N_Positive_Pairs": int((df["Label"] == 1).sum()),
        "N_Negative_Pairs": int((df["Label"] == 0).sum()),

        "Macro_AUC0.1": macro_auc01,
        "Macro_AUPRC": macro_auprc,

        "Pooled_AUC0.1": pooled_auc01,
        "Pooled_AUPRC": pooled_auprc,
    }


# ============================================================
# Main
# ============================================================

def main():

    conn = sqlite3.connect(DB_PATH)

    overall_results = []
    all_mutation_results = []

    for model in MODEL_TABLES:

        print(f"Processing {model} ...")

        query = f"""
        SELECT
            ID,
            Peptide,
            HLA,
            CDR3a,
            CDR3b,
            Va,
            Ja,
            Vb,
            Jb,
            Label,
            log2foldchange,
            TCR_Group,
            Epitope_Group,
            Prediction_Prob
        FROM "{model}"
        WHERE
            Peptide IS NOT NULL
            AND Label IS NOT NULL
            AND Prediction_Prob IS NOT NULL
        """

        raw_df = pd.read_sql_query(
            query,
            conn
        )

        df = prepare_data(raw_df)

        # -----------------------------
        # Per-mutation metrics
        # -----------------------------

        mutation_df = calculate_per_mutation_metrics(df)
        mutation_df.insert(0, "Model", model)

        all_mutation_results.append(
            mutation_df
        )

        # -----------------------------
        # Overall metrics
        # -----------------------------

        overall = calculate_overall_metrics(
            df,
            mutation_df
        )

        overall["Model"] = model
        overall_results.append(overall)

    conn.close()

    # ========================================================
    # Overall summary
    # ========================================================

    overall_df = pd.DataFrame(
        overall_results
    )

    overall_df = overall_df[
        [
            "Model",
            "N_Pairs",
            "N_Mutations",
            "N_Eligible_Mutations",
            "N_Positive_Pairs",
            "N_Negative_Pairs",
            "Macro_AUC0.1",
            "Macro_AUPRC",
            "Pooled_AUC0.1",
            "Pooled_AUPRC",
        ]
    ]

    overall_df = overall_df.sort_values(
        "Macro_AUC0.1",
        ascending=False
    )

    print("\n========================================")
    print("Experiment 1: Overall mutation prediction")
    print("========================================\n")

    print(
        overall_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    overall_df.to_csv(
        "experiment1_overall_metrics.csv",
        index=False
    )

    # ========================================================
    # Per-mutation detail
    # ========================================================

    mutation_df = pd.concat(
        all_mutation_results,
        ignore_index=True
    )

    mutation_df = mutation_df.sort_values(
        [
            "Model",
            "Mutation_Position",
            "Mutation",
        ]
    )

    mutation_df.to_csv(
        "experiment1_per_mutation_metrics.csv",
        index=False
    )

    print(
        "\nSaved:\n"
        "  experiment1_overall_metrics.csv\n"
        "  experiment1_per_mutation_metrics.csv"
    )


if __name__ == "__main__":
    main()
