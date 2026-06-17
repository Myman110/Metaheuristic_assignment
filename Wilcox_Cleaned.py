import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

FILES = {"Baseline": "Baseline_paper_results.xlsx", "ALNS_AOS": "ALNS_AOS_Results.xlsx", "Hybrid_GA_ALNS": "hybrid_ph_mh_alns_seed_results.xlsx", "ALNS_Q_learning": "ALNS_q_learning.xlsx"}

outfile = "wilcoxon_case_means_runtime_ARPD_no_paper100.xlsx"

KEYS = ["table", "case", "seed"]

fit_col = {"Baseline": "MH_fitness","ALNS_AOS": "ALNS_fitness","ALNS_Q_learning": "ALNS_fitness","Hybrid_GA_ALNS": "Hybrid_fitness"}

time_col = {"Baseline": "MH_runtime","ALNS_AOS": "ALNS_runtime","ALNS_Q_learning": "ALNS_runtime","Hybrid_GA_ALNS": "Hybrid_ALNS_runtime"}

paper_df = pd.DataFrame([
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


def clean_sheet(df, method, sheet_name, col_dict):
    df = df.copy()
    req_col = col_dict[method]

    if "BaseCase" in df.columns:
        df["case"] = df["BaseCase"].astype(str)
    elif "case_file" in df.columns:
        df["case"] = (df["case_file"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0])
    elif "n" in df.columns:
        df["case"] = df["n"].astype(str)
    else:
        raise ValueError("No case column found")

    if "table" in df.columns:
        df["table"] = (df["table"].astype(str).str.lower().str.replace(" ", "", regex=False).str.replace("_", "", regex=False))
    elif "table8" in sheet_name.lower() or "n" in df.columns:
        df["table"] = "table8"
    elif "table14" in sheet_name.lower() or "basecase" in [c.lower() for c in df.columns]:
        df["table"] = "table14"
    else:
        df["table"] = "unknown"

    out = df[["table", "case", "seed", req_col]].copy()
    out = out.rename(columns={req_col: method})
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out[method] = pd.to_numeric(out[method], errors="coerce")
    out = out.dropna(subset=["seed", method])
    out["seed"] = out["seed"].astype(int)
    return out


def read_file(path, method, col_dict):
    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        if "seed" not in sheet.lower():
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        if df.empty:
            continue
        try:
            frames.append(clean_sheet(df, method, sheet, col_dict))
        except Exception as e:
            print(f"Skipping {method} / {sheet}: {e}")
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(subset=KEYS, keep="first")


def build_wide(col_dict):
    method_data = {m: read_file(fp, m, col_dict)for m, fp in FILES.items()}
    for m, df in method_data.items():
        print(f"{m}: {len(df)} rows loaded")

    dfs = list(method_data.values())
    wide = dfs[0]
    for df in dfs[1:]:
        wide = wide.merge(df, on=KEYS, how="outer")
    return wide


def sort_cases(df):
    t8 = df[df["table"] == "table8"].copy()
    t14 = df[df["table"] == "table14"].copy()
    if not t8.empty:
        t8["sort"] = pd.to_numeric(t8["case"], errors="coerce")
        t8 = t8.sort_values("sort").drop(columns="sort")
    if not t14.empty:
        order = ["2M38", "2M46", "6M140", "6M163"]
        t14["sort"] = t14["case"].apply(lambda x: order.index(x) if x in order else 999)
        t14 = t14.sort_values("sort").drop(columns="sort")
    return pd.concat([t8, t14], ignore_index=True)


def calc_means(df, suffix):
    rows = []
    for table in sorted(df["table"].dropna().unique()):
        tdf = df[df["table"] == table]
        for case in sorted(tdf["case"].dropna().unique()):
            cdf = tdf[tdf["case"] == case]
            row = {"table": table,"case": str(case),"N_seeds": cdf["seed"].nunique()}
            for method in FILES:
                row[f"{method}_{suffix}"] = cdf[method].mean()
            rows.append(row)
    return sort_cases(pd.DataFrame(rows))

fit_wide = build_wide(fit_col)
results = []

for m1, m2 in combinations(FILES.keys(), 2):
    paired = fit_wide[["table", "case", "seed", m1, m2]].dropna()
    if paired.empty:
        results.append({
            "Comparison": f"{m1} vs {m2}",
            "Method_1": m1,
            "Method_2": m2,
            "N_pairs": 0,
            "Wilcoxon_statistic": np.nan,
            "p_value": np.nan,
            "Significant_at_0.05": False,
            "Note": "No paired observations"})
        continue

    x = paired[m1]
    y = paired[m2]
    diff = y - x

    if (diff == 0).all():
        stat = 0
        p_val = 1.0
        note = "All differences are zero"

    else:
        stat, p_val = wilcoxon(x, y)
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
        "p_value": p_val,
        "Significant_at_0.05": p_val < 0.05,
        "Note": note})

wilcox_df = pd.DataFrame(results)
fit_df = calc_means(fit_wide, "mean_fitness")
time_wide = build_wide(time_col)
time_df = calc_means(time_wide, "mean_time")

summary = fit_df.merge(time_df,on=["table", "case", "N_seeds"],how="outer")

summary = summary.merge(paper_df,on=["table", "case"],how="left")

for method in FILES:
    fit_name = f"{method}_mean_fitness"
    summary[f"{method}_ARPD_vs_Paper_%"] = np.where(summary["Paper_result"] == 0, np.nan, ((summary[fit_name] - summary["Paper_result"])/ summary["Paper_result"]) * 100)

summary = sort_cases(summary)
with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
    wilcox_df.to_excel(writer,sheet_name="wilcoxon_pairwise",index=False)

    summary.to_excel(writer,sheet_name="case_summary",index=False)

    fit_df.to_excel(writer,sheet_name="mean_fitness",index=False)

    time_df.to_excel(writer, sheet_name="mean_runtime",index=False)

    paper_df.to_excel(writer, sheet_name="paper_results",index=False)

    fit_wide.to_excel(writer, sheet_name="paired_seed_data", index=False)

print(f"Saved: {outfile}")