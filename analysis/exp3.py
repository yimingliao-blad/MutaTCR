import sqlite3
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr


# ============================================================
# Configuration
# ============================================================

# --- made path-configurable for the staged pipeline (originally DB_PATH = "fp.db") ---
import os as _os
from pathlib import Path as _Path
DB_PATH = _os.environ.get("TCRJ_DB", "fp.db")
_OUT = _Path(_os.environ.get("TCRJ_OUT", "."))
_OUT.mkdir(parents=True, exist_ok=True)
_os.chdir(_OUT)   # the script writes its CSVs relative to the working directory
DB_PATH = str(_Path(DB_PATH).resolve()) if _Path(DB_PATH).exists() else DB_PATH
REFERENCE_PEPTIDE = "YLQPRTFLL"

TABLES = {
    "ERGO": "ERGO",
    "ERGO2": "ERGO2",
    "NetTCR": "NetTCR",
    "NetTCR2.2": "NetTCR22",
    "TITAN": "TITAN",
    "EPACT": "EPACT",
    "PanPep": "PanPep",
    "SCEPTR": "SCEPTR",
}

MODEL_ORDER = [
    "ERGO",
    "EPACT",
    "PanPep",
    "TITAN",
    "NetTCR",
    "NetTCR2.2",
    "ERGO2",
    "SCEPTR",
]


# ============================================================
# Helper functions
# ============================================================

def get_mutation_info(peptide, reference=REFERENCE_PEPTIDE):
    """
    Identify a single amino-acid substitution relative to reference.

    Returns
    -------
    position : int
        1-based mutation position.
    wt : str
        Wild-type residue.
    mutant : str
        Mutant residue.
    mutation : str
        Mutation label, e.g. R5K.
    """

    if pd.isna(peptide):
        return np.nan, np.nan, np.nan, np.nan

    peptide = str(peptide).strip()

    if peptide == reference:
        return np.nan, np.nan, np.nan, "Reference"

    if len(peptide) != len(reference):
        raise ValueError(
            f"Peptide length mismatch: {peptide} vs {reference}"
        )

    diffs = [
        i
        for i, (ref_residue, peptide_residue)
        in enumerate(zip(reference, peptide))
        if ref_residue != peptide_residue
    ]

    if len(diffs) != 1:
        raise ValueError(
            f"Expected one substitution, found {len(diffs)}: {peptide}"
        )

    i = diffs[0]

    position = i + 1
    wt = reference[i]
    mutant = peptide[i]
    mutation = f"{wt}{position}{mutant}"

    return position, wt, mutant, mutation


def identify_tcr_column(df):
    """
    Choose a stable TCR identifier.
    """

    if "TCR_Group" in df.columns:
        return "TCR_Group"

    if "CDR3b" in df.columns:
        return "CDR3b"

    if "ID" in df.columns:
        return "ID"

    raise ValueError(
        "Could not identify a TCR identifier column."
    )


def convert_numeric_column(df, column_name, model_name):
    """
    Convert a dataframe column to numeric and report invalid entries.
    """

    if column_name not in df.columns:
        raise ValueError(
            f"{model_name}: missing required column '{column_name}'."
        )

    original_non_null = df[column_name].notna()

    df[column_name] = pd.to_numeric(
        df[column_name],
        errors="coerce"
    )

    invalid = original_non_null & df[column_name].isna()
    n_invalid = int(invalid.sum())

    if n_invalid > 0:
        print(
            f"WARNING: {model_name}: "
            f"{n_invalid} values in {column_name} "
            f"could not be converted to numeric."
        )

    return df


# ============================================================
# Load database and process each model
# ============================================================

conn = sqlite3.connect(DB_PATH)

all_auc_rows = []
experimental_df = None

for model in MODEL_ORDER:

    table = TABLES[model]

    print(f"\nProcessing {model} from table {table}...")

    df = pd.read_sql_query(
        f"SELECT * FROM {table}",
        conn
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "Peptide",
        "log2foldchange",
        "Prediction_Prob",
    ]

    for column_name in required_columns:
        if column_name not in df.columns:
            raise ValueError(
                f"{model}: required column '{column_name}' "
                f"is missing."
            )

    # --------------------------------------------------------
    # Explicit numeric conversion
    # --------------------------------------------------------

    df = convert_numeric_column(
        df,
        "log2foldchange",
        model
    )

    df = convert_numeric_column(
        df,
        "Prediction_Prob",
        model
    )

    tcr_col = identify_tcr_column(df)

    # --------------------------------------------------------
    # Reference peptide rows
    # --------------------------------------------------------

    ref = df[
        df["Peptide"].astype(str).str.strip()
        == REFERENCE_PEPTIDE
    ].copy()

    if ref.empty:
        raise ValueError(
            f"{model}: reference peptide "
            f"{REFERENCE_PEPTIDE} not found."
        )

    ref_values = (
        ref[
            [
                tcr_col,
                "log2foldchange",
            ]
        ]
        .dropna(
            subset=[
                tcr_col,
                "log2foldchange",
            ]
        )
        .drop_duplicates(
            subset=[tcr_col]
        )
        .rename(
            columns={
                "log2foldchange":
                    "Reference_log2foldchange"
            }
        )
    )

    # --------------------------------------------------------
    # Mutant peptide rows
    # --------------------------------------------------------

    mut = df[
        df["Peptide"].astype(str).str.strip()
        != REFERENCE_PEPTIDE
    ].copy()

    mut = mut.merge(
        ref_values,
        on=tcr_col,
        how="left"
    )

    mut["Reference_log2foldchange"] = pd.to_numeric(
        mut["Reference_log2foldchange"],
        errors="coerce"
    )

    n_missing_reference = int(
        mut["Reference_log2foldchange"].isna().sum()
    )

    if n_missing_reference > 0:
        print(
            f"WARNING: {model}: "
            f"{n_missing_reference} mutant rows "
            f"do not have a valid reference measurement."
        )

    # --------------------------------------------------------
    # Experimental mutation effect
    # --------------------------------------------------------

    mut["Delta_log2FC"] = (
        mut["log2foldchange"]
        - mut["Reference_log2foldchange"]
    )

    # Experimental binary label
    mut["Experimental_Label"] = pd.to_numeric(
        mut["Label"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Mutation annotations
    # --------------------------------------------------------

    mutation_info = mut["Peptide"].apply(
        get_mutation_info
    )

    mutation_info_df = pd.DataFrame(
        mutation_info.tolist(),
        index=mut.index,
        columns=[
            "Mutation_Position",
            "WT_Residue",
            "Mutant_Residue",
            "Mutation",
        ]
    )

    mut[
        [
            "Mutation_Position",
            "WT_Residue",
            "Mutant_Residue",
            "Mutation",
        ]
    ] = mutation_info_df

    # --------------------------------------------------------
    # Save one copy of experimental measurements
    # --------------------------------------------------------

    if experimental_df is None:

        experimental_df = mut[
            [
                tcr_col,
                "Peptide",
                "Mutation",
                "Mutation_Position",
                "WT_Residue",
                "Mutant_Residue",
                "log2foldchange",
                "Reference_log2foldchange",
                "Delta_log2FC",
                "Experimental_Label",
            ]
        ].copy()

    # --------------------------------------------------------
    # Mutation-specific AUC01
    # --------------------------------------------------------

    for mutation, group in mut.groupby(
        "Mutation",
        dropna=True
    ):

        g = group.dropna(
            subset=[
                "Experimental_Label",
                "Prediction_Prob",
            ]
        ).copy()

        if g.empty:
            continue

        y = g["Experimental_Label"].astype(int).values
        scores = g["Prediction_Prob"].astype(float).values

        # ROC AUC requires both experimental classes
        if np.unique(y).size < 2:
            continue

        auc01 = roc_auc_score(
            y,
            scores,
            max_fpr=0.1
        )

        first = g.iloc[0]

        all_auc_rows.append(
            {
                "Model": model,
                "Mutation": mutation,
                "Mutation_Position":
                    int(first["Mutation_Position"]),
                "WT_Residue":
                    first["WT_Residue"],
                "Mutant_Residue":
                    first["Mutant_Residue"],
                "AUC01":
                    float(auc01),
                "N_TCR":
                    int(len(g)),
                "N_Positive":
                    int(y.sum()),
                "N_Negative":
                    int(len(y) - y.sum()),
            }
        )


conn.close()


# ============================================================
# Validate experimental dataframe
# ============================================================

if experimental_df is None:
    raise RuntimeError(
        "No experimental data were extracted."
    )


# ============================================================
# Experimental mutation severity
# ============================================================

severity_rows = []

for mutation, group in experimental_df.groupby(
    "Mutation",
    dropna=True
):

    g = group.dropna(
        subset=[
            "Mutation_Position",
            "Delta_log2FC",
            "Experimental_Label",
        ]
    ).copy()

    if g.empty:
        continue

    first = g.iloc[0]

    mean_delta_log2fc = float(
        g["Delta_log2FC"].mean()
    )

    # Larger value means stronger loss of recognition
    experimental_disruption = -mean_delta_log2fc

    binder_fraction = float(
        g["Experimental_Label"].mean()
    )

    binder_loss_fraction = (
        1.0 - binder_fraction
    )

    severity_rows.append(
        {
            "Mutation":
                mutation,

            "Mutation_Position":
                int(first["Mutation_Position"]),

            "WT_Residue":
                first["WT_Residue"],

            "Mutant_Residue":
                first["Mutant_Residue"],

            "Mean_Delta_log2FC":
                mean_delta_log2fc,

            "Experimental_Disruption":
                experimental_disruption,

            "Binder_Fraction":
                binder_fraction,

            "Binder_Loss_Fraction":
                binder_loss_fraction,

            "Mean_Mutant_log2FC":
                float(
                    g["log2foldchange"].mean()
                ),

            "Mean_Reference_log2FC":
                float(
                    g[
                        "Reference_log2foldchange"
                    ].mean()
                ),

            "N_TCR":
                int(len(g)),
        }
    )

severity_df = pd.DataFrame(
    severity_rows
)


# ============================================================
# Mutation-specific AUC dataframe
# ============================================================

auc_df = pd.DataFrame(
    all_auc_rows
)

if auc_df.empty:
    raise RuntimeError(
        "No mutation-specific AUC values were calculated."
    )


# ============================================================
# Merge severity with prediction performance
# ============================================================

merged = auc_df.merge(
    severity_df,
    on=[
        "Mutation",
        "Mutation_Position",
        "WT_Residue",
        "Mutant_Residue",
    ],
    how="left"
)

merged["Position"] = (
    "P"
    + merged["Mutation_Position"].astype(str)
)

merged["R5_Status"] = np.where(
    merged["Mutation_Position"] == 5,
    "R5",
    "non-R5"
)

merged["Model"] = pd.Categorical(
    merged["Model"],
    categories=MODEL_ORDER,
    ordered=True
)

merged = merged.sort_values(
    [
        "Model",
        "Mutation_Position",
        "Mutation",
    ]
).reset_index(drop=True)


# ============================================================
# Per-model correlation analysis
# ============================================================

correlation_rows = []

for model in MODEL_ORDER:

    g = merged[
        merged["Model"] == model
    ].copy()

    g = g.dropna(
        subset=[
            "Experimental_Disruption",
            "Binder_Loss_Fraction",
            "AUC01",
        ]
    )

    if len(g) < 3:
        print(
            f"WARNING: {model} has too few valid "
            f"mutations for correlation analysis."
        )
        continue

    # --------------------------------------------------------
    # Continuous disruption vs AUC01
    # --------------------------------------------------------

    spearman_disruption = spearmanr(
        g["Experimental_Disruption"],
        g["AUC01"]
    )

    pearson_disruption = pearsonr(
        g["Experimental_Disruption"],
        g["AUC01"]
    )

    # --------------------------------------------------------
    # Binder-loss fraction vs AUC01
    # --------------------------------------------------------

    spearman_binder_loss = spearmanr(
        g["Binder_Loss_Fraction"],
        g["AUC01"]
    )

    correlation_rows.append(
        {
            "Model":
                model,

            "N_Mutations":
                int(len(g)),

            "Spearman_Rho_Disruption":
                float(
                    spearman_disruption.statistic
                ),

            "Spearman_P_Disruption":
                float(
                    spearman_disruption.pvalue
                ),

            "Pearson_R_Disruption":
                float(
                    pearson_disruption.statistic
                ),

            "Pearson_P_Disruption":
                float(
                    pearson_disruption.pvalue
                ),

            "Spearman_Rho_BinderLoss":
                float(
                    spearman_binder_loss.statistic
                ),

            "Spearman_P_BinderLoss":
                float(
                    spearman_binder_loss.pvalue
                ),
        }
    )

corr_df = pd.DataFrame(
    correlation_rows
)


# ============================================================
# Cross-model mutation summary
# ============================================================

mutation_cross_model = (
    merged
    .groupby(
        [
            "Mutation",
            "Mutation_Position",
            "WT_Residue",
            "Mutant_Residue",
            "Experimental_Disruption",
            "Binder_Fraction",
            "Binder_Loss_Fraction",
        ],
        as_index=False,
        observed=True
    )
    .agg(
        Mean_AUC01=(
            "AUC01",
            "mean"
        ),
        Median_AUC01=(
            "AUC01",
            "median"
        ),
        SD_AUC01=(
            "AUC01",
            "std"
        ),
        N_Models=(
            "AUC01",
            "count"
        ),
    )
)


# ============================================================
# Cross-model correlations
# ============================================================

cross_model_valid = mutation_cross_model.dropna(
    subset=[
        "Experimental_Disruption",
        "Binder_Loss_Fraction",
        "Mean_AUC01",
    ]
).copy()

cross_spearman_disruption = spearmanr(
    cross_model_valid[
        "Experimental_Disruption"
    ],
    cross_model_valid[
        "Mean_AUC01"
    ]
)

cross_spearman_binder_loss = spearmanr(
    cross_model_valid[
        "Binder_Loss_Fraction"
    ],
    cross_model_valid[
        "Mean_AUC01"
    ]
)

cross_pearson_disruption = pearsonr(
    cross_model_valid[
        "Experimental_Disruption"
    ],
    cross_model_valid[
        "Mean_AUC01"
    ]
)


# ============================================================
# Print summary
# ============================================================

print("\n========================================")
print("Cross-model mutation analysis")
print("========================================")

print(
    "Experimental disruption vs mean AUC01:"
)

print(
    f"  Spearman rho = "
    f"{cross_spearman_disruption.statistic:.3f}"
)

print(
    f"  p = "
    f"{cross_spearman_disruption.pvalue:.4g}"
)

print(
    f"  Pearson r = "
    f"{cross_pearson_disruption.statistic:.3f}"
)

print(
    f"  p = "
    f"{cross_pearson_disruption.pvalue:.4g}"
)

print()

print(
    "Binder-loss fraction vs mean AUC01:"
)

print(
    f"  Spearman rho = "
    f"{cross_spearman_binder_loss.statistic:.3f}"
)

print(
    f"  p = "
    f"{cross_spearman_binder_loss.pvalue:.4g}"
)


print("\n========================================")
print("Per-model correlations")
print("========================================")

display_columns = [
    "Model",
    "N_Mutations",
    "Spearman_Rho_Disruption",
    "Spearman_P_Disruption",
    "Spearman_Rho_BinderLoss",
    "Spearman_P_BinderLoss",
]

print(
    corr_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# Optional R5 vs non-R5 summary
# ============================================================

r5_summary = (
    merged
    .groupby(
        [
            "Model",
            "R5_Status",
        ],
        as_index=False,
        observed=True
    )
    .agg(
        Mean_AUC01=(
            "AUC01",
            "mean"
        ),
        Median_AUC01=(
            "AUC01",
            "median"
        ),
        SD_AUC01=(
            "AUC01",
            "std"
        ),
        N_Mutations=(
            "AUC01",
            "count"
        ),
    )
)

print("\n========================================")
print("R5 vs non-R5 summary")
print("========================================")

print(
    r5_summary.to_string(
        index=False
    )
)


# ============================================================
# Save outputs
# ============================================================

severity_df.to_csv(
    "mutation_experimental_severity.csv",
    index=False
)

auc_df.to_csv(
    "mutation_specific_auc.csv",
    index=False
)

merged.to_csv(
    "severity_vs_prediction_auc.csv",
    index=False
)

corr_df.to_csv(
    "severity_auc_correlations_by_model.csv",
    index=False
)

mutation_cross_model.to_csv(
    "severity_vs_cross_model_auc.csv",
    index=False
)

r5_summary.to_csv(
    "r5_vs_nonr5_auc_summary.csv",
    index=False
)


# ============================================================
# Also save a compact plotting file
# ============================================================

plot_df = mutation_cross_model[
    [
        "Mutation",
        "Mutation_Position",
        "WT_Residue",
        "Mutant_Residue",
        "Experimental_Disruption",
        "Binder_Fraction",
        "Binder_Loss_Fraction",
        "Mean_AUC01",
        "Median_AUC01",
        "SD_AUC01",
        "N_Models",
    ]
].copy()

plot_df["Position"] = (
    "P"
    + plot_df[
        "Mutation_Position"
    ].astype(str)
)

plot_df.to_csv(
    "fig_severity_vs_mean_auc.csv",
    index=False
)


print("\nGenerated files:")
print(
    "  mutation_experimental_severity.csv"
)
print(
    "  mutation_specific_auc.csv"
)
print(
    "  severity_vs_prediction_auc.csv"
)
print(
    "  severity_auc_correlations_by_model.csv"
)
print(
    "  severity_vs_cross_model_auc.csv"
)
print(
    "  r5_vs_nonr5_auc_summary.csv"
)
print(
    "  fig_severity_vs_mean_auc.csv"
)
