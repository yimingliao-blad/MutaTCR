import sqlite3
import pandas as pd
import numpy as np

# --- made path-configurable for the staged pipeline (originally DB_PATH = "fp.db") ---
import os as _os
from pathlib import Path as _Path
DB_PATH = _os.environ.get("TCRJ_DB", "fp.db")
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


def get_mutation_info(peptide, reference=REFERENCE_PEPTIDE):
    """
    Infer mutation information relative to YLQPRTFLL.

    Returns:
        Mutation
        Mutation_Position
        WT_Residue
        Mutant_Residue
    """

    if not isinstance(peptide, str):
        return None, None, None, None

    peptide = peptide.strip()

    if peptide == reference:
        return "REF", 0, None, None

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


# ============================================================
# Read all 8 tables
# ============================================================

conn = sqlite3.connect(DB_PATH)

all_models = []

for model in MODEL_TABLES:

    print(f"Reading {model} ...")

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
    """

    df = pd.read_sql_query(query, conn)

    df.insert(0, "Model", model)

    all_models.append(df)

conn.close()

combined = pd.concat(
    all_models,
    ignore_index=True
)


# ============================================================
# Clean fields
# ============================================================

combined["Peptide"] = (
    combined["Peptide"]
    .astype(str)
    .str.strip()
)

combined["Label"] = pd.to_numeric(
    combined["Label"],
    errors="coerce"
)

combined["log2foldchange"] = pd.to_numeric(
    combined["log2foldchange"],
    errors="coerce"
)

combined["Prediction_Prob"] = pd.to_numeric(
    combined["Prediction_Prob"],
    errors="coerce"
)

combined = combined.dropna(
    subset=[
        "Peptide",
        "Label",
        "log2foldchange",
        "Prediction_Prob",
    ]
)

combined["Label"] = combined["Label"].astype(int)


# ============================================================
# Mutation information
# ============================================================

mutation_info = combined["Peptide"].apply(
    get_mutation_info
)

combined["Mutation"] = mutation_info.apply(
    lambda x: x[0]
)

combined["Mutation_Position"] = mutation_info.apply(
    lambda x: x[1]
)

combined["WT_Residue"] = mutation_info.apply(
    lambda x: x[2]
)

combined["Mutant_Residue"] = mutation_info.apply(
    lambda x: x[3]
)

combined["Is_Reference"] = (
    combined["Peptide"] == REFERENCE_PEPTIDE
)

# Remove malformed/unexpected peptides
combined = combined[
    combined["Mutation"].notna()
].copy()


# ============================================================
# Reference experimental log2FC for every Model x TCR
# ============================================================

reference_exp = (
    combined[
        combined["Is_Reference"]
    ]
    .groupby(
        ["Model", "TCR_Group"],
        as_index=False
    )
    .agg(
        Reference_log2foldchange=(
            "log2foldchange",
            "mean"
        )
    )
)

combined = combined.merge(
    reference_exp,
    on=["Model", "TCR_Group"],
    how="left"
)

combined["Delta_log2FC"] = (
    combined["log2foldchange"]
    - combined["Reference_log2foldchange"]
)


# ============================================================
# Reference model score for every Model x TCR
#
# This will be useful later when comparing predicted mutation
# effects with experimentally observed mutation effects.
# ============================================================

reference_pred = (
    combined[
        combined["Is_Reference"]
    ]
    .groupby(
        ["Model", "TCR_Group"],
        as_index=False
    )
    .agg(
        Reference_Prediction_Prob=(
            "Prediction_Prob",
            "mean"
        )
    )
)

combined = combined.merge(
    reference_pred,
    on=["Model", "TCR_Group"],
    how="left"
)

combined["Delta_Prediction"] = (
    combined["Prediction_Prob"]
    - combined["Reference_Prediction_Prob"]
)


# ============================================================
# Mutation rows only
# ============================================================

mut_df = combined[
    ~combined["Is_Reference"]
].copy()


# ============================================================
# Check experimental consistency across model tables
#
# For the same TCR x peptide pair, Label and log2foldchange
# should normally be identical across tables.
# ============================================================

consistency = (
    mut_df
    .groupby(
        [
            "TCR_Group",
            "Peptide",
            "Mutation",
        ]
    )
    .agg(
        N_Models=("Model", "nunique"),
        N_Label_Values=("Label", "nunique"),
        N_log2FC_Values=("log2foldchange", "nunique"),
        Min_log2FC=("log2foldchange", "min"),
        Max_log2FC=("log2foldchange", "max"),
    )
    .reset_index()
)

inconsistent = consistency[
    (consistency["N_Label_Values"] > 1)
    |
    (consistency["N_log2FC_Values"] > 1)
]

print("\nExperimental consistency check:")
print(f"Total TCR-mutant pairs: {len(consistency)}")
print(
    f"Inconsistent experimental records: "
    f"{len(inconsistent)}"
)

if len(inconsistent) > 0:
    inconsistent.to_csv(
        "experimental_inconsistencies.csv",
        index=False
    )
    print(
        "Saved inconsistencies to "
        "experimental_inconsistencies.csv"
    )


# ============================================================
# Unified all-model pair-level file
# ============================================================

combined_output = mut_df[
    [
        "Model",
        "ID",
        "TCR_Group",
        "Peptide",
        "Mutation",
        "Mutation_Position",
        "WT_Residue",
        "Mutant_Residue",
        "HLA",
        "CDR3a",
        "CDR3b",
        "Va",
        "Ja",
        "Vb",
        "Jb",
        "Label",
        "log2foldchange",
        "Reference_log2foldchange",
        "Delta_log2FC",
        "Prediction_Prob",
        "Reference_Prediction_Prob",
        "Delta_Prediction",
        "Epitope_Group",
    ]
].copy()

combined_output = combined_output.sort_values(
    [
        "Model",
        "Mutation_Position",
        "Mutant_Residue",
        "TCR_Group",
    ]
)

combined_output.to_csv(
    "all_models_mutation_pair_level.csv",
    index=False
)


# ============================================================
# Experimental mutation summary
#
# Since experimental data should be identical across model
# tables, first deduplicate Model.
# ============================================================

experimental_unique = (
    mut_df[
        [
            "TCR_Group",
            "Peptide",
            "Mutation",
            "Mutation_Position",
            "WT_Residue",
            "Mutant_Residue",
            "Label",
            "log2foldchange",
            "Delta_log2FC",
        ]
    ]
    .drop_duplicates(
        subset=[
            "TCR_Group",
            "Peptide",
        ]
    )
)


mutation_summary = (
    experimental_unique
    .groupby(
        [
            "Peptide",
            "Mutation",
            "Mutation_Position",
            "WT_Residue",
            "Mutant_Residue",
        ],
        as_index=False
    )
    .agg(
        N_TCR=("TCR_Group", "nunique"),
        N_Positive=("Label", "sum"),

        Mean_log2FC=(
            "log2foldchange",
            "mean"
        ),

        Median_log2FC=(
            "log2foldchange",
            "median"
        ),

        Mean_Delta_log2FC=(
            "Delta_log2FC",
            "mean"
        ),

        Median_Delta_log2FC=(
            "Delta_log2FC",
            "median"
        ),
    )
)

mutation_summary["Binder_Fraction"] = (
    mutation_summary["N_Positive"]
    / mutation_summary["N_TCR"]
)

mutation_summary = mutation_summary.sort_values(
    [
        "Mutation_Position",
        "Mutant_Residue",
    ]
)

mutation_summary.to_csv(
    "experiment2_mutation_summary.csv",
    index=False
)


# ============================================================
# Position-level experimental summary
# ============================================================

position_summary = (
    mutation_summary
    .groupby(
        "Mutation_Position",
        as_index=False
    )
    .agg(
        N_Mutations=(
            "Mutation",
            "count"
        ),

        Mean_Binder_Fraction=(
            "Binder_Fraction",
            "mean"
        ),

        Median_Binder_Fraction=(
            "Binder_Fraction",
            "median"
        ),

        SD_Binder_Fraction=(
            "Binder_Fraction",
            "std"
        ),

        Mean_log2FC=(
            "Mean_log2FC",
            "mean"
        ),

        Mean_Delta_log2FC=(
            "Mean_Delta_log2FC",
            "mean"
        ),

        Median_Delta_log2FC=(
            "Median_Delta_log2FC",
            "median"
        ),
    )
)

position_summary.to_csv(
    "experiment2_position_summary.csv",
    index=False
)


# ============================================================
# R5 versus non-R5
# ============================================================

mutation_summary["R5_Group"] = np.where(
    mutation_summary["Mutation_Position"] == 5,
    "R5",
    "non-R5"
)

r5_summary = (
    mutation_summary
    .groupby(
        "R5_Group",
        as_index=False
    )
    .agg(
        N_Mutations=(
            "Mutation",
            "count"
        ),

        Mean_Binder_Fraction=(
            "Binder_Fraction",
            "mean"
        ),

        Median_Binder_Fraction=(
            "Binder_Fraction",
            "median"
        ),

        Mean_Delta_log2FC=(
            "Mean_Delta_log2FC",
            "mean"
        ),

        Median_Delta_log2FC=(
            "Median_Delta_log2FC",
            "median"
        ),
    )
)

r5_summary.to_csv(
    "experiment2_r5_summary.csv",
    index=False
)


# ============================================================
# Mutation ordering for figures
# ============================================================

mutation_order = (
    mutation_summary[
        [
            "Mutation",
            "Mutation_Position",
            "Mutant_Residue",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "Mutation_Position",
            "Mutant_Residue",
        ]
    )
    .reset_index(drop=True)
)

mutation_order["x"] = (
    np.arange(len(mutation_order)) + 1
)


# ============================================================
# Figure 4a
# ============================================================

fig4a = mutation_summary.merge(
    mutation_order,
    on=[
        "Mutation",
        "Mutation_Position",
        "Mutant_Residue",
    ],
    how="left"
)

fig4a = fig4a.sort_values("x")

fig4a.to_csv(
    "fig4a_mutation_binder_fraction.csv",
    index=False
)


# ============================================================
# Position blocks for P1-P9
# ============================================================

position_blocks = (
    mutation_order
    .groupby(
        "Mutation_Position"
    )
    .agg(
        x_start=("x", "min"),
        x_end=("x", "max"),
        n_mutations=("x", "size"),
    )
    .reset_index()
)

position_blocks["x_center"] = (
    position_blocks["x_start"]
    + position_blocks["x_end"]
) / 2

position_blocks.to_csv(
    "fig4_position_blocks.csv",
    index=False
)


# ============================================================
# Experimental TCR x mutation heatmap
# ============================================================

tcr_order = sorted(
    experimental_unique["TCR_Group"]
    .dropna()
    .unique()
)

tcr_to_index = {
    tcr: i + 1
    for i, tcr in enumerate(tcr_order)
}

fig4d = experimental_unique.merge(
    mutation_order,
    on=[
        "Mutation",
        "Mutation_Position",
        "Mutant_Residue",
    ],
    how="left"
)

fig4d["tcr_index"] = (
    fig4d["TCR_Group"]
    .map(tcr_to_index)
)

fig4d = fig4d[
    [
        "x",
        "tcr_index",
        "TCR_Group",
        "Mutation",
        "Mutation_Position",
        "Label",
        "log2foldchange",
        "Delta_log2FC",
    ]
].sort_values(
    [
        "tcr_index",
        "x",
    ]
)

fig4d.to_csv(
    "fig4d_experimental_heatmap.csv",
    index=False
)


# ============================================================
# Model-level mutation summaries
#
# Useful for the NEXT experiment:
# compare model-predicted mutation effects against experiment.
# ============================================================

model_mutation_summary = (
    mut_df
    .groupby(
        [
            "Model",
            "Peptide",
            "Mutation",
            "Mutation_Position",
            "WT_Residue",
            "Mutant_Residue",
        ],
        as_index=False
    )
    .agg(
        N_TCR=("TCR_Group", "nunique"),

        Mean_Prediction=(
            "Prediction_Prob",
            "mean"
        ),

        Median_Prediction=(
            "Prediction_Prob",
            "median"
        ),

        Mean_Delta_Prediction=(
            "Delta_Prediction",
            "mean"
        ),

        Median_Delta_Prediction=(
            "Delta_Prediction",
            "median"
        ),

        Mean_Experimental_Delta=(
            "Delta_log2FC",
            "mean"
        ),
    )
)

model_mutation_summary.to_csv(
    "all_models_mutation_effect_summary.csv",
    index=False
)


# ============================================================
# Output
# ============================================================

print("\n=== Position summary ===")
print(
    position_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\n=== R5 vs non-R5 ===")
print(
    r5_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("\nGenerated files:")
print("  all_models_mutation_pair_level.csv")
print("  all_models_mutation_effect_summary.csv")
print("  experiment2_mutation_summary.csv")
print("  experiment2_position_summary.csv")
print("  experiment2_r5_summary.csv")
print("  fig4a_mutation_binder_fraction.csv")
print("  fig4_position_blocks.csv")
print("  fig4d_experimental_heatmap.csv")
