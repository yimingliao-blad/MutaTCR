# TCR-Antigen Binding Classification Rules

## Dataset: TetTCR-SeqHD (TCR_pMHC_annotated.csv)

### Overview

This document describes the binary classification rules for determining positive TCR-antigen binding from the TetTCR-SeqHD dataset. The classification is based on UMI (Unique Molecular Identifier) counts across 282 antigen columns.

---

## Key Columns

| Column | Description |
|--------|-------------|
| `specificity.final` | Human-curated binding assignment (single antigen, multi-antigen list, NEG, or filter) |
| `SignalRatio` | Sum of UMI for assigned antigens / Total UMI (correlation with calculated: 0.9816) |
| `threshold15` | Alternative classification (different threshold) |
| Columns 38-319 | Antigen UMI counts (282 antigens) |

---

## Classification Categories

### 1. Single-Antigen Cases (High Confidence)
- **Format**: Single antigen name (e.g., `EBV-BRLF1`, `M1`)
- **Criteria**: Top antigen ratio >= 0.4
- **Count**: 15,232 total / 5,224 unique TCR-antigen pairs
- **Validation**: 100% have signal ratio >= 0.4

### 2. Multi-Antigen Cases (Cross-Reactive Binding)
- **Format**: Pipe-separated list (e.g., `HCV-K1Y|HCV-K1S|HCV-L2I|...`)
- **Criteria**: Top antigen ratio < 0.4, but multiple antigens show binding signal
- **Count**: 4,081 cases
- **Key Finding**: Antigens are ordered by UMI count (94.7% first antigen matches calculated top)

### 3. NEG (Negative)
- **Criteria**: No clear binding signal above threshold
- **SignalRatio**: Mean 0.254, Median 0.165

### 4. Filter (Excluded)
- **Criteria**: Ambiguous or low-quality signal
- **SignalRatio**: Mean 0.396, Median 0.330

---

## Binary Classification Rule

### Rule 1: Signal Ratio Threshold (Single-Antigen)

```
IF (antigen_UMI / total_UMI) >= 0.4:
    THEN positive_binding = TRUE
    THEN binding_antigen = top_antigen
```

**Validation Results:**
- 99.22% of single-antigen cases have ratio > 0.4
- Minimum ratio in single-antigen cases: exactly 0.4

### Rule 2: Population Frequency (Multi-Antigen)

```
IF (antigen_UMI / total_UMI) < 0.4:
    IF top_antigen appears in >= 10 unique TCRs across dataset:
        THEN positive_binding = TRUE (confirmed by population frequency)
    ELSE:
        THEN positive_binding = UNCERTAIN (possible cross-reactivity)
```

**Validation Results:**
- 16 antigens appear as top in 10+ unique TCRs with ratio < 0.4
- 1,848 records (87.6% of low-ratio cases) confirmed by this rule

---

## Signal Ratio Distribution by Category

| Category | Mean | Median | Count |
|----------|------|--------|-------|
| Single-antigen | 0.746 | 0.770 | 15,232 |
| Multi-antigen | 0.895 | 0.930 | 4,081 |
| Filter | 0.396 | 0.330 | ~3,000 |
| NEG | 0.254 | 0.165 | ~1,500 |

---

## Multi-Antigen Characteristics

### Distinguishing Features

| Metric | Single-Antigen | Multi-Antigen |
|--------|----------------|---------------|
| Max ratio (median) | 0.767 | 0.283 |
| Ratio gap to 2nd (median) | 0.667 | 0.053 |
| Total UMI (median) | 17 | 49 |

### Common Cross-Reactive Sets

| Antigen Set | Frequency |
|-------------|-----------|
| MART1-A27L \| MART1-ALA | 617 |
| MART1-26-35 \| MART1-A27L \| MART1-ALA | 598 |
| HCV-K1S \| HCV-K1Y \| HCV-K1YI7V \| HCV-L2I \| HCVNS3-1406-1415 | 519 |
| MART1-ALA \| PGT-178 | 203 |
| PB1-590-599 \| PB1-591-599 | 185 |

---

## Open Questions for Further Investigation

### Potential Unified Indicator

The current binary classification uses two separate rules. A unified indicator might combine:

1. **Signal Ratio**: Direct binding strength measure
2. **Ratio Gap**: Difference between top and second antigen (selectivity)
3. **Population Frequency**: How often the antigen appears as top across unique TCRs
4. **Total UMI**: Overall signal quality/depth

### Candidate Unified Metrics to Explore

```
Option A: Weighted Score
score = w1 * signal_ratio + w2 * ratio_gap + w3 * log(population_freq)

Option B: Confidence-Adjusted Ratio
adjusted_ratio = signal_ratio * (1 + log(population_freq) / max_log_freq)

Option C: Bayesian Posterior
P(binding | ratio, freq) proportional to P(ratio | binding) * P(freq | binding)
```

### Questions to Investigate

1. Is there a single threshold on a combined metric that separates positive from negative?
2. Can we predict `specificity.final` category from UMI data alone?
3. What is the biological meaning of cross-reactive antigen sets (e.g., MART1 variants)?
4. How does `threshold15` differ from `specificity.final`?

---

## Summary

| Binding Type | Primary Criterion | Secondary Criterion | Confidence |
|--------------|-------------------|---------------------|------------|
| Single-antigen | ratio >= 0.4 | - | High |
| Multi-antigen (frequent) | ratio < 0.4 | top_antigen in 10+ TCRs | Medium-High |
| Multi-antigen (rare) | ratio < 0.4 | top_antigen in < 10 TCRs | Low |
| NEG/Filter | ratio < 0.4 | no consistent pattern | Negative |

---

## Investigation: Unified Indicator

### Hypothesis 1: Average Ratio Across Same TCR

For multi-antigen cases, grouping by TCR (CDR3 beta) and averaging the signal ratio across records:

| Metric | Binding Antigens | Non-Binding Antigens |
|--------|------------------|----------------------|
| Count | 1,739 | 2,739 |
| Mean ratio | 0.2319 | 0.0353 |
| Median ratio | 0.2115 | 0.0225 |
| Std | 0.1694 | 0.0381 |

**Ratio of means: 6.57x** (binding vs non-binding)

#### Threshold Analysis (TCR-averaged ratio)

| Threshold | Binding Above | Non-Binding Above | Separation |
|-----------|---------------|-------------------|------------|
| 0.05 | 84.24% | 22.86% | 61.38% |
| 0.10 | 70.96% | 5.66% | 65.30% |
| 0.15 | 61.87% | 2.37% | 59.50% |
| 0.20 | 51.70% | 0.88% | 50.82% |
| 0.25 | 41.58% | 0.11% | 41.47% |

**Conclusion**: Threshold of 0.10 provides good separation (71% binding captured, only 5.7% false positives)

---

### Hypothesis 2: clone0 as Background Reference

#### clone0 Characteristics

- **NOT background noise** - it's a well-characterized TCR clone
- Single TCR sequence: `CASSFLGTGLNEQYF` (421 cells)
- Highly specific for HCV antigens (HCVNS3, K1Y, L2I, K1S, K1YI7V)
- Higher SignalRatio (mean 0.877) than other clones (mean 0.601)

#### clone0 UMI Count Analysis

| Metric | Binding (5 HCV) | Non-Binding (277 others) |
|--------|-----------------|--------------------------|
| Mean UMI per antigen per cell | **9.45** | **0.043** |
| Total UMI per cell | 47.27 | 11.85 |
| Non-zero rate | 99%+ | 2.1% |
| **Ratio (Binding/Non-Binding)** | | **221x** |

#### Individual Binding Antigens (HCV)

| Antigen | Mean UMI | Median UMI | Non-zero Rate |
|---------|----------|------------|---------------|
| HCVNS3.1406.1415 | 11.43 | 9.0 | 99.0% |
| HCV.K1Y | 11.15 | 10.0 | 99.3% |
| HCV.L2I | 10.20 | 9.0 | 99.8% |
| HCV.K1S | 9.02 | 9.0 | 99.3% |
| HCV.K1YI7V | 5.47 | 5.0 | 96.9% |

#### Background Noise Distribution (Non-Binding)

| UMI Count | Frequency |
|-----------|-----------|
| 0 | 97.90% |
| 1 | 1.55% |
| 2 | 0.28% |
| 3+ | 0.27% |

**Key Finding**: Non-binding signal is mostly zero (97.9%), with occasional 1-2 UMI noise. True binding shows consistent signal (mean ~9-11 UMI) across cells.

#### Background Noise Ratio Statistics

| Metric | Value |
|--------|-------|
| Mean ratio | 0.000387 |
| 99th percentile ratio | 0.0152 |
| Max ratio | 0.2179 |

**Using 99th percentile (0.0152) as threshold: 98.62% of binding antigens pass**

---

### Generalized Background Analysis (All Multi-Antigen Cases)

#### Per-Row Ratio Distribution

| Metric | Binding | Non-Binding |
|--------|---------|-------------|
| Mean | 0.2253 | 0.0368 |
| Median | 0.2000 | 0.0256 |
| 95th percentile | - | 0.1034 |
| 99th percentile | - | 0.1786 |

#### Optimal Threshold (Per-Row)

| Threshold | Binding Above | Non-Binding Above | Separation |
|-----------|---------------|-------------------|------------|
| 0.03 | 95.04% | 41.93% | 53.11% |
| 0.05 | 89.70% | 21.39% | 68.31% |
| 0.06 | 87.71% | 16.33% | **71.39%** |
| 0.08 | 82.82% | 9.14% | **73.68%** |
| 0.10 | 76.97% | 5.07% | 71.90% |

**Best threshold: 0.06-0.08** (maximum separation ~74%)

---

## Final Classification Rules (Corrected)

After investigation, using only signal ratio >= 0.4 leads to false positives from low-UMI noise.

**Analysis of "extra" positives (high ratio but not in specificity.final):**
- 1,302 cases had avg_signal_ratio >= 0.4 but NOT in specificity.final
- 98% had total_umi < 10 (inflated ratio due to low denominator)
- avg_umi median: 1.0 (vs 10.0 for true positives)

**Conclusion:** Only use human-curated `specificity.final` for positives.

### Final Rules

```python
# Strong positive: ONLY human-curated annotation
strong_positive = (in_specificity_final == 1)

# Strong negative: low UMI and not positive
strong_negative = (avg_umi < 3) AND (strong_positive == 0)

# Possible cross-reactive: uncertain cases
possible_cross_reactive = NOT strong_positive AND NOT strong_negative
```

### Validation Results

| Category | Count | Percentage |
|----------|-------|------------|
| Strong Positive | 6,221 | 11.0% |
| Strong Negative | 45,318 | 79.9% |
| Possible Cross-Reactive | 5,217 | 9.2% |
| **Total** | **56,756** | 100% |

### Why Old Dataset Had Only 350 Positives

The old processing script used Kevin's "TCRs from Tetramer Signals" sheet which contains only **281 curated TCRs**. The new approach uses the full `TCR_pMHC_annotated.csv` with **32,992 rows (6,841 unique valid TCRs)**.

---

## Next Steps for Investigation

1. Validate unified threshold across different antigen types (HCV vs MART1 vs EBV)
2. Incorporate total UMI count as quality weight
3. Test if ratio_gap adds predictive power
4. ROC analysis to find optimal unified threshold

---

*Document created: 2024*
*Data source: TetTCR-SeqHD dataset*
*Analysis based on first 1000 rows (validated) and full dataset (32,992 rows)*
