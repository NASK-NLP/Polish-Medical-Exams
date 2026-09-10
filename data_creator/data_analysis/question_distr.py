import pandas as pd 
from pathlib import Path
from config.paths import DATA_DIR, MODEL_OUTPUT, DATA_CREATOR, FIG_DIR
import pickle 
import os
from data_creator.utils import load_hf_split_compatible

from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import math


def spread_autopct_labels(autotexts, min_gap=0.12):
    """Spread percentage labels vertically per side to reduce overlaps."""
    if not autotexts:
        return

    positioned = []
    for t in autotexts:
        x, y = t.get_position()
        if t.get_text().strip() == "":
            continue
        # Keep each label roughly on its original circular track.
        radius = math.sqrt(x * x + y * y)
        side = "right" if x >= 0 else "left"
        positioned.append({"text": t, "x": x, "y": y, "radius": radius, "side": side})

    for side in ("left", "right"):
        side_items = [item for item in positioned if item["side"] == side]
        if not side_items:
            continue

        side_items.sort(key=lambda item: item["y"])
        for i in range(1, len(side_items)):
            if side_items[i]["y"] - side_items[i - 1]["y"] < min_gap:
                side_items[i]["y"] = side_items[i - 1]["y"] + min_gap

        for item in side_items:
            y_new = max(min(item["y"], item["radius"]), -item["radius"])
            x_new = math.sqrt(max(item["radius"] ** 2 - y_new ** 2, 0))
            if side == "left":
                x_new *= -1
            item["text"].set_position((x_new, y_new))


def reorder_pie_slices(group_df):
    """Order slices to avoid clustering tiny adjacent wedges.

    Strategy: sort by size descending, then alternate from the largest and
    smallest remaining slices to distribute small wedges around the ring.
    """
    ordered = group_df.sort_values("count", ascending=False).reset_index(drop=True)
    left = 0
    right = len(ordered) - 1
    pick_large = True
    index_order = []

    while left <= right:
        if pick_large:
            index_order.append(left)
            left += 1
        else:
            index_order.append(right)
            right -= 1
        pick_large = not pick_large

    return ordered.iloc[index_order].reset_index(drop=True)

dataset_names = ["r_lek", "r_ldek", "r_pes_latest", "r_diagnostics", "r_pharmacy","r_lek_en", "r_ldek_en"]

q_id_dict = {}
for method in DATA_DIR.iterdir():
    if method.is_dir():
        print(f"Directory: {method.name}")
        if "r_" not in method.name:
            method_qid_dict = {}
            temp_path = DATA_DIR / method.name
            for file in temp_path.iterdir():
                if "gt" in file.name:
                    file_path = temp_path / file.name
                    temp_df = pd.read_csv(file_path)

                    temp_df["uuid"] = temp_df["uuid"].astype(str)

                    print(f"Len of method df {method.name}: {len(temp_df)}")
                    ids = set(temp_df["uuid"])
                    method_qid_dict[file.name[:-6]] = ids
            q_id_dict[method.name] = method_qid_dict

method_qid_dict = {}
for data_id in dataset_names:
    data = load_hf_split_compatible(data_id)

    data["uuid"] = data["uuid"].astype(str)

    method_qid_dict[data_id] = set(data["uuid"])

q_id_dict["original"] = method_qid_dict

save_path = DATA_CREATOR / "data_analysis"
save_path.mkdir(parents=True, exist_ok=True)



count_data = []
for q_type, sources in q_id_dict.items():
    for source, ids in sources.items():
        count_data.append({
            "question_type": q_type,
            "source": source,
            "count": len(ids)
        })

counts_df = pd.DataFrame(count_data)
counts_df['source'] = pd.Categorical(counts_df['source'], categories=dataset_names, ordered=True)
counts_df = counts_df.sort_values('source')
print("=== Counts per source per question type ===")
print(counts_df)

f_dir = FIG_DIR / "composition"
os.makedirs(f_dir, exist_ok=True)

color_map = {
    "Not covered": "#4E79A7",               # deep blue
    "Multiple Statements": "#F28E2B",       # orange
    "Multiple Answers": "#FFBE7D",          # lighter orange, related
    "Correct answer substitution": "#52A145", # teal
    "Open-ended generation": "#E15759"      # coral/red
}
name_mapping = {"r_lek":"LEK", "r_ldek": "LDEK", "r_pes_latest": "PES", "r_diagnostics": "PESDL" , "r_pharmacy": "PESF", "r_lek_en": "LEK EN", "r_ldek_en": "LDEK EN"}
for source, group_df in counts_df.groupby('source'):
    group_df = group_df.copy()
    sum_others = group_df.loc[group_df["question_type"] != "original", "count"].sum()
    group_df.loc[group_df["question_type"] == "original", "count"] -= sum_others
    group_df.loc[group_df["question_type"] == "original", "question_type"] = "Not covered"
    group_df.loc[group_df["question_type"] == "multiple_choice", "question_type"] = "Multiple Statements"
    group_df.loc[group_df["question_type"] == "multiple_choice2","question_type" ]= "Multiple Answers"
    group_df.loc[group_df["question_type"] == "abstaining_substitution", "question_type"] ="Correct answer substitution"
    group_df.loc[group_df["question_type"] == "free_form","question_type"]=  "Open-ended generation"
    group_df = reorder_pie_slices(group_df)
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = [color_map[q] for q in group_df['question_type']]
    wedges, texts, autotexts = ax.pie(
        group_df['count'],
        labels=None,  
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=1.25, 
        colors = colors,
        wedgeprops={'width': 0.4, 'edgecolor': 'black'},
        textprops={'color': 'black', 'weight': 'bold', 'fontsize': 12}
    )
    spread_autopct_labels(autotexts, min_gap=0.14)

    ax.legend(
        group_df['question_type'],
        title="Question Type",
        loc="center left",
        bbox_to_anchor=(0.9, 1)
    )
    ax.text(
        0, 0, name_mapping[source],
        ha='center', va='center',
        fontsize=18, fontweight='bold'
    )
    plt.tight_layout()

    fpath = f_dir / f"composition_{source}.pdf"
    plt.savefig(fpath, format="pdf")
    plt.close(fig)
sources = counts_df['source'].unique()
n_sources = len(sources)
# 2-row layout
n_rows = 2
n_cols = math.ceil(n_sources / n_rows)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(6 * n_cols, 6 * n_rows),
    gridspec_kw={"wspace": 0.02, "hspace": 0.08},
    subplot_kw=dict(aspect="equal")
)

# Flatten axes for easy iteration
axes = axes.flatten()

legend_labels = None
legend_colors = None
wedges_for_legend = None

for ax, source in zip(axes, sources):
    group_df = counts_df[counts_df['source'] == source].copy()

    sum_others = group_df.loc[group_df["question_type"] != "original", "count"].sum()
    group_df.loc[group_df["question_type"] == "original", "count"] -= sum_others

    group_df["question_type"] = group_df["question_type"].replace({
        "original": "Not covered",
        "multiple_choice": "Multiple Statements",
        "multiple_choice2": "Multiple Answers",
        "abstaining_substitution": "Correct answer substitution",
        "free_form": "Open-ended generation"
    })
    group_df = reorder_pie_slices(group_df)

    colors = [color_map[q] for q in group_df['question_type']]

    wedges, _, autotexts = ax.pie(
        group_df['count'],
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=1.23,
        colors=colors,
        wedgeprops={'width': 0.4, 'edgecolor': 'black'},
        textprops={'color': 'black', 'fontsize': 20}
    )
    spread_autopct_labels(autotexts, min_gap=0.2)

    ax.text(
        0, 0,
        name_mapping[source],
        ha='center', va='center',
        fontsize=25
    )

    # Store legend info once
    if legend_labels is None:
        legend_labels = group_df['question_type'].tolist()
        wedges_for_legend = wedges

# ---- Legend in the empty (8th) subplot ----
legend_ax = axes[len(sources)]
legend_ax.axis("off")

legend_ax.legend(
    wedges_for_legend,
    legend_labels,
    title="Question Type",
    loc="center",
    title_fontsize=24,
    fontsize=20,
    ncol=1,
    frameon=False
)

fig.subplots_adjust(left=0.03, right=0.98, top=0.98, bottom=0.03, wspace=0.02, hspace=0.08)

# Save as SVG
plt.savefig(f_dir / "composition_all_sources.pdf", format="pdf")
plt.close(fig)
coverage_data = []
sum_o_c = 0
sum_c_c = 0
for source in dataset_names:
    if source not in q_id_dict["original"]:
        continue

    original_ids = q_id_dict["original"][source]
    other_ids = set()

    # Combine all UUIDs from other question types for this source
    for q_type, sources in q_id_dict.items():
        if q_type == "original":
            continue
        if source in sources:
            other_ids.update(sources[source])
    covered = len(original_ids.intersection(other_ids))
    total = len(original_ids)
    coverage_ratio = covered / total if total > 0 else 0

    coverage_data.append({
        "source": name_mapping[source],
        "original_count": total,
        "covered_count": covered,
        "coverage_ratio": round(coverage_ratio,2)
    })
    sum_o_c += total
    sum_c_c += covered
 
coverage_data.append({
        "source": "All",
        "original_count": sum_o_c,
        "covered_count": sum_c_c,
        "coverage_ratio":  round(sum_c_c/sum_o_c,2)
    })
coverage_df = pd.DataFrame(coverage_data)
coverage_df.to_csv(save_path/"distr.csv")
print("\n=== Coverage of original datasets by all other question types ===")
print(coverage_df)


