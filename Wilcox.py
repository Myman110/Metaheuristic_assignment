import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

# =========================
# INPUT FILES
# =========================

files = {
    "Baseline": "Baseline_paper_results.xlsx",
    "ALNS_AOS": "alns_aos_table8_table14_seed_results.xlsx",
    "Hybrid_GA_ALNS": "hybrid_ph_mh_alns_seed_results.xlsx",
    "ALNS_Q_learning": "ALNS_q_learning.xlsx",
}

output_file = "wilcoxon_case_means_runtime_ARPD_no_paper100.xlsx"

# =========================
# COLUMN RULES
# =========================

fitness_column = {
    "Baseline": "MH_fitness",
    "ALNS_AOS": "ALNS_fitness",
    "ALNS_Q_learning": "ALNS_fitness",
    "Hybrid_GA_ALNS": "Hybrid_fitness",
}

runtime_column = {
    "Baseline": "MH_runtime",
    "ALNS_AOS": "ALNS_runtime",
    "ALNS_Q_learning": "ALNS_runtime",
    "Hybrid_GA_ALNS": "Hybrid_ALNS_runtime",
}

# =========================
# REAL PAPER RESULTS
# =========================

paper_results = pd.DataFrame([
    ["table8", "15", 10, 0],
    ["table8", "25", 10, 0],
    ["table8", "30", 10, 0],
    ["table8", "60", 10, 10],
    ["table8", "90", 10, 22.2],
    ["table8", "120", 10, 37.47],
    ["table8", "140", 10, 45.6],
    ["table14", "2M38", 10, 10.1],
    ["table14", "2M46", 10, 17.1],
    ["table14", "6M140", 10, 45.6],
    ["table14", "6M163", 10, 59.5],
], columns=["table", "case", "N_seeds_paper", "Paper_result"])

# =========================
# HELPER FUNCTIONS
# =========================

def normalize_instance_name(series):
    return (
        series.astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[0]
    )


def normalize_table_name(series):
    return (
        series.astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )


def normalize_sheet(df, method, sheet_name, column_dict):
    df = df.copy()
    required_col = column_dict[method]

    if required_col not in df.columns:
        raise ValueError(
            f"{method} requires column '{required_col}', "
            f"available columns: {list(df.columns)}"
        )

    if "seed" not in df.columns:
        raise ValueError("No seed column found")

    if "BaseCase" in df.columns:
        df["case"] = df["BaseCase"].astype(str)
    elif "case_file" in df.columns:
        df["case"] = normalize_instance_name(df["case_file"])
    elif "n" in df.columns:
        df["case"] = df["n"].astype(str)
    else:
        raise ValueError("No case column found")

    if "table" in df.columns:
        df["table"] = normalize_table_name(df["table"])
    elif "table8" in sheet_name.lower() or "n" in df.columns:
        df["table"] = "table8"
    elif "table14" in sheet_name.lower() or "basecase" in [c.lower() for c in df.columns]:
        df["table"] = "table14"
    else:
        df["table"] = "unknown"

    out = df[["table", "case", "seed", required_col]].copy()
    out = out.rename(columns={required_col: method})

    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out[method] = pd.to_numeric(out[method], errors="coerce")

    out = out.dropna(subset=["seed", method])
    out["seed"] = out["seed"].astype(int)

    return out


def read_method_file(file_path, method, column_dict):
    xls = pd.ExcelFile(file_path)
    frames = []

    for sheet in xls.sheet_names:
        if "seed" not in sheet.lower():
            continue

        df = pd.read_excel(file_path, sheet_name=sheet)

        if df.empty:
            continue

        try:
            frames.append(normalize_sheet(df, method, sheet, column_dict))
        except Exception as e:
            print(f"Skipping {method} / {sheet}: {e}")

    if not frames:
        raise ValueError(f"No usable seed-level sheets found for {method}")

    result = pd.concat(frames, ignore_index=True)

    result = result.drop_duplicates(
        subset=["table", "case", "seed"],
        keep="first"
    )

    return result


def build_wide_dataset(column_dict):
    method_data = {}

    for method, file_path in files.items():
        method_data[method] = read_method_file(file_path, method, column_dict)
        print(f"{method}: {len(method_data[method])} rows loaded")

    wide = None

    for method, df in method_data.items():
        if wide is None:
            wide = df
        else:
            wide = wide.merge(
                df,
                on=["table", "case", "seed"],
                how="outer"
            )

    return wide


def sort_cases(df):
    table8_df = df[df["table"] == "table8"].copy()
    table14_df = df[df["table"] == "table14"].copy()

    if not table8_df.empty:
        table8_df["case_sort"] = pd.to_numeric(table8_df["case"], errors="coerce")
        table8_df = table8_df.sort_values("case_sort").drop(columns="case_sort")

    if not table14_df.empty:
        order = ["2M38", "2M46", "6M140", "6M163"]
        table14_df["case_sort"] = table14_df["case"].apply(
            lambda x: order.index(x) if x in order else 999
        )
        table14_df = table14_df.sort_values("case_sort").drop(columns="case_sort")

    return pd.concat([table8_df, table14_df], ignore_index=True)


# =========================
# FITNESS DATA
# =========================

fitness_wide = build_wide_dataset(fitness_column)

# =========================
# WILCOXON TESTS
# =========================

results = []

for m1, m2 in combinations(files.keys(), 2):

    paired = fitness_wide[["table", "case", "seed", m1, m2]].dropna()

    if paired.empty:
        results.append({
            "Comparison": f"{m1} vs {m2}",
            "Method_1": m1,
            "Method_2": m2,
            "N_pairs": 0,
            "Wilcoxon_statistic": np.nan,
            "p_value": np.nan,
            "Significant_at_0.05": False,
            "Note": "No paired observations"
        })
        continue

    x = paired[m1]
    y = paired[m2]
    diff = y - x

    if (diff == 0).all():
        stat = 0
        p_value = 1.0
        note = "All differences are zero"
    else:
        stat, p_value = wilcoxon(x, y, alternative="two-sided")
        note = ""

    results.append({
        "Comparison": f"{m1} vs {m2}",
        "Method_1": m1,
        "Method_2": m2,
        "N_pairs": len(paired),
        f"{m1}_mean": x.mean(),
        f"{m2}_mean": y.mean(),
        "Mean_difference_Method2_minus_Method1": diff.mean(),
        "Median_difference_Method2_minus_Method1": diff.median(),
        "Wilcoxon_statistic": stat,
        "p_value": p_value,
        "Significant_at_0.05": p_value < 0.05,
        "Note": note
    })

results_df = pd.DataFrame(results)

# =========================
# CASE FITNESS AVERAGES
# =========================

case_rows = []

for table_name in sorted(fitness_wide["table"].dropna().unique()):
    table_df = fitness_wide[fitness_wide["table"] == table_name]

    for case in sorted(table_df["case"].dropna().unique()):
        case_df = table_df[table_df["case"] == case]

        row = {
            "table": table_name,
            "case": str(case),
            "N_seeds": case_df["seed"].nunique()
        }

        for method in files.keys():
            row[f"{method}_mean_fitness"] = case_df[method].mean()

        case_rows.append(row)

case_averages_df = sort_cases(pd.DataFrame(case_rows))

# =========================
# MEAN COMPUTATION TIME
# =========================

runtime_wide = build_wide_dataset(runtime_column)

time_rows = []

for table_name in sorted(runtime_wide["table"].dropna().unique()):
    table_df = runtime_wide[runtime_wide["table"] == table_name]

    for case in sorted(table_df["case"].dropna().unique()):
        case_df = table_df[table_df["case"] == case]

        row = {
            "table": table_name,
            "case": str(case),
            "N_seeds": case_df["seed"].nunique()
        }

        for method in files.keys():
            row[f"{method}_mean_time"] = case_df[method].mean()

        time_rows.append(row)

runtime_averages_df = sort_cases(pd.DataFrame(time_rows))

# =========================
# COMBINED SUMMARY WITH PAPER + ARPD
# =========================

summary_df = case_averages_df.merge(
    runtime_averages_df,
    on=["table", "case", "N_seeds"],
    how="outer"
)

summary_df = summary_df.merge(
    paper_results,
    on=["table", "case"],
    how="left"
)

methods_for_arpd = [
    "Baseline",
    "ALNS_AOS",
    "Hybrid_GA_ALNS",
    "ALNS_Q_learning",
]

for method in methods_for_arpd:
    method_col = f"{method}_mean_fitness"

    summary_df[f"{method}_ARPD_vs_Paper_%"] = np.where(
        summary_df["Paper_result"] == 0,
        np.nan,
        ((summary_df[method_col] - summary_df["Paper_result"]) / summary_df["Paper_result"]) * 100
    )

summary_df = sort_cases(summary_df)

# =========================
# SAVE OUTPUT
# =========================

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="wilcoxon_pairwise", index=False)
    summary_df.to_excel(writer, sheet_name="case_summary", index=False)
    case_averages_df.to_excel(writer, sheet_name="mean_fitness", index=False)
    runtime_averages_df.to_excel(writer, sheet_name="mean_runtime", index=False)
    paper_results.to_excel(writer, sheet_name="paper_results", index=False)
    fitness_wide.to_excel(writer, sheet_name="paired_seed_data", index=False)

print(f"Saved: {output_file}")