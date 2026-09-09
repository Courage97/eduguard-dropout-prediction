"""
=======================================================================
Dataset Visualisation — Distribution Charts and Graphs
Dropout Risk Prediction System
Nzeako Jennifer Nkiruka | DU/C/247
=======================================================================
What this script generates:
  1.  Overall dropout distribution (pie chart)
  2.  Dropout rate by gender
  3.  Dropout rate by household income band
  4.  Dropout rate by fee payment status
  5.  Dropout rate by parental involvement
  6.  Dropout rate by parental education level
  7.  Attendance rate distribution (histogram)
  8.  Average academic score distribution (histogram)
  9.  Dropout rate by learning pace (cognitive)
  10. Dropout rate by PE participation (psychomotor)
  11. Cognitive vs academic score scatter plot
  12. Dropout rate by distance to school (binned)
  13. Correlation heatmap of all features
  14. Feature-by-feature dropout rate comparison (summary bar)

All charts saved to:  assets/charts/
High resolution:      300 DPI — suitable for thesis documentation

Run:  python visualise.py
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────────────
RAW_DATA  = "data/dropout_dataset_10000.csv"
CHART_DIR = "assets/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
PURPLE     = "#4B3FA0"
TEAL       = "#0F6E56"
RED        = "#991B1B"
AMBER      = "#B45309"
LIGHT_GRAY = "#F5F5F3"
WHITE      = "#FFFFFF"

DROPOUT_COLORS    = [TEAL, RED]      # 0=Completed, 1=Dropout
CATEGORY_PALETTE  = [PURPLE, TEAL, RED, AMBER, "#1E3A5F", "#2D6A4F"]

# ── Matplotlib global style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.titlecolor":    PURPLE,
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     WHITE,
    "figure.facecolor":   WHITE,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  WHITE,
})

print("=" * 65)
print("  Dataset Visualisation Script")
print("  Dropout Risk Prediction System — Jennifer Nkiruka")
print("=" * 65)

# ── Load raw dataset ───────────────────────────────────────────────────────────
df = pd.read_csv(RAW_DATA)
print(f"\n  Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Dropout rate:   {df['Dropout_Status'].mean()*100:.1f}%")
print(f"  Saving charts to: {CHART_DIR}/\n")

chart_count = 0

def save_chart(fig, filename, title):
    global chart_count
    chart_count += 1
    path = f"{CHART_DIR}/{filename}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [{chart_count:02d}] Saved: {filename}  — {title}")


# ══════════════════════════════════════════════════════════════════════
# CHART 1 — Overall Dropout Distribution (Pie)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))

counts  = df["Dropout_Status"].value_counts().sort_index()
labels  = ["Completed Year\n(0)", "Dropped Out\n(1)"]
colors  = [TEAL, RED]
explode = (0.03, 0.07)

wedges, texts, autotexts = ax.pie(
    counts,
    labels=labels,
    colors=colors,
    explode=explode,
    autopct="%1.1f%%",
    startangle=140,
    textprops={"fontsize": 12},
    wedgeprops={"linewidth": 1.5, "edgecolor": WHITE},
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
    at.set_color(WHITE)

ax.set_title(
    "Figure 1: Overall Dropout Status Distribution\n"
    f"(N = {len(df):,} students)",
    pad=20
)

fig.text(
    0.5, 0.01,
    f"Dropout: {counts[1]:,} students ({counts[1]/len(df)*100:.1f}%)  |  "
    f"Completed: {counts[0]:,} students ({counts[0]/len(df)*100:.1f}%)",
    ha="center", fontsize=10, color="#555555"
)
save_chart(fig, "01_dropout_distribution_pie.png",
           "Overall dropout distribution")


# ══════════════════════════════════════════════════════════════════════
# CHART 2 — Dropout Rate by Gender
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: count chart
gender_counts = df.groupby(["Gender", "Dropout_Status"]).size().unstack()
gender_counts.plot(
    kind="bar", ax=axes[0],
    color=DROPOUT_COLORS, edgecolor=WHITE, linewidth=0.8,
    width=0.6
)
axes[0].set_title("Student Count by Gender and Dropout Status")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Number of Students")
axes[0].set_xticklabels(["Female", "Male"], rotation=0)
axes[0].legend(["Completed (0)", "Dropped Out (1)"], frameon=False)
for p in axes[0].patches:
    axes[0].annotate(
        f"{int(p.get_height()):,}",
        (p.get_x() + p.get_width() / 2, p.get_height() + 30),
        ha="center", fontsize=9, color="#333333"
    )

# Right: dropout rate
gender_rate = df.groupby("Gender")["Dropout_Status"].mean() * 100
bars = axes[1].bar(
    gender_rate.index, gender_rate.values,
    color=[RED if g == "Female" else PURPLE for g in gender_rate.index],
    edgecolor=WHITE, linewidth=0.8, width=0.5
)
axes[1].set_title("Dropout Rate (%) by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Dropout Rate (%)")
axes[1].set_ylim(0, 50)
axes[1].axhline(
    df["Dropout_Status"].mean() * 100,
    color=AMBER, linestyle="--", linewidth=1.5, label="Overall average"
)
axes[1].legend(frameon=False)
for bar, rate in zip(bars, gender_rate.values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=12
    )

fig.suptitle(
    "Figure 2: Gender and Dropout Risk\n"
    "(Source: Adeyemo et al., 2023 — rural female students at 2.3× higher risk)",
    fontsize=12, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "02_dropout_by_gender.png", "Dropout rate by gender")


# ══════════════════════════════════════════════════════════════════════
# CHART 3 — Dropout Rate by Household Income Band
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

income_order = ["Low", "Medium", "High"]
income_counts = df.groupby(
    ["Household_Income_Band", "Dropout_Status"]
).size().unstack().reindex(income_order)

income_counts.plot(
    kind="bar", ax=axes[0],
    color=DROPOUT_COLORS, edgecolor=WHITE, linewidth=0.8, width=0.6
)
axes[0].set_title("Student Count by Income Band")
axes[0].set_xlabel("Household Income Band")
axes[0].set_ylabel("Number of Students")
axes[0].set_xticklabels(income_order, rotation=0)
axes[0].legend(["Completed (0)", "Dropped Out (1)"], frameon=False)

income_rate = (
    df.groupby("Household_Income_Band")["Dropout_Status"]
    .mean() * 100
).reindex(income_order)

bar_colors = [RED, AMBER, TEAL]
bars = axes[1].bar(
    income_rate.index, income_rate.values,
    color=bar_colors, edgecolor=WHITE, linewidth=0.8, width=0.5
)
axes[1].set_title("Dropout Rate (%) by Income Band")
axes[1].set_xlabel("Household Income Band")
axes[1].set_ylabel("Dropout Rate (%)")
axes[1].set_ylim(0, 65)
axes[1].axhline(
    df["Dropout_Status"].mean() * 100,
    color=PURPLE, linestyle="--", linewidth=1.5, label="Overall average"
)
axes[1].legend(frameon=False)
for bar, rate in zip(bars, income_rate.values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=12
    )

fig.suptitle(
    "Figure 3: Household Income Band and Dropout Risk\n"
    "(Source: World Bank GHS-Panel Nigeria — low income households show 60% lower completion)",
    fontsize=12, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "03_dropout_by_income.png", "Dropout rate by income band")


# ══════════════════════════════════════════════════════════════════════
# CHART 4 — Dropout Rate by Fee Payment Status
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

fee_order = ["Unpaid", "Partial", "Paid"]
fee_counts = df.groupby(
    ["Fee_Payment_Status", "Dropout_Status"]
).size().unstack().reindex(fee_order)

fee_counts.plot(
    kind="bar", ax=axes[0],
    color=DROPOUT_COLORS, edgecolor=WHITE, linewidth=0.8, width=0.6
)
axes[0].set_title("Student Count by Fee Payment Status")
axes[0].set_xlabel("Fee Payment Status")
axes[0].set_ylabel("Number of Students")
axes[0].set_xticklabels(fee_order, rotation=0)
axes[0].legend(["Completed (0)", "Dropped Out (1)"], frameon=False)

fee_rate = (
    df.groupby("Fee_Payment_Status")["Dropout_Status"]
    .mean() * 100
).reindex(fee_order)

bar_colors = [RED, AMBER, TEAL]
bars = axes[1].bar(
    fee_rate.index, fee_rate.values,
    color=bar_colors, edgecolor=WHITE, linewidth=0.8, width=0.5
)
axes[1].set_title("Dropout Rate (%) by Fee Payment Status")
axes[1].set_xlabel("Fee Payment Status")
axes[1].set_ylabel("Dropout Rate (%)")
axes[1].set_ylim(0, 80)
axes[1].axhline(
    df["Dropout_Status"].mean() * 100,
    color=PURPLE, linestyle="--", linewidth=1.5, label="Overall average"
)
axes[1].legend(frameon=False)
for bar, rate in zip(bars, fee_rate.values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=12
    )

fig.suptitle(
    "Figure 4: Fee Payment Status and Dropout Risk\n"
    "(Source: Ojo & Awonuga, 2022 — unpaid fees primary dropout reason in 38% of cases)",
    fontsize=12, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "04_dropout_by_fee_status.png",
           "Dropout rate by fee payment status")


# ══════════════════════════════════════════════════════════════════════
# CHART 5 — Dropout Rate by Parental Involvement
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

inv_order = ["Absent", "Passive", "Active"]
inv_counts = df.groupby(
    ["Parental_Involvement", "Dropout_Status"]
).size().unstack().reindex(inv_order)

inv_counts.plot(
    kind="bar", ax=axes[0],
    color=DROPOUT_COLORS, edgecolor=WHITE, linewidth=0.8, width=0.6
)
axes[0].set_title("Student Count by Parental Involvement")
axes[0].set_xlabel("Parental Involvement Level")
axes[0].set_ylabel("Number of Students")
axes[0].set_xticklabels(inv_order, rotation=0)
axes[0].legend(["Completed (0)", "Dropped Out (1)"], frameon=False)

inv_rate = (
    df.groupby("Parental_Involvement")["Dropout_Status"]
    .mean() * 100
).reindex(inv_order)

bar_colors = [RED, AMBER, TEAL]
bars = axes[1].bar(
    inv_rate.index, inv_rate.values,
    color=bar_colors, edgecolor=WHITE, linewidth=0.8, width=0.5
)
axes[1].set_title("Dropout Rate (%) by Parental Involvement")
axes[1].set_xlabel("Parental Involvement Level")
axes[1].set_ylabel("Dropout Rate (%)")
axes[1].set_ylim(0, 60)
axes[1].axhline(
    df["Dropout_Status"].mean() * 100,
    color=PURPLE, linestyle="--", linewidth=1.5, label="Overall average"
)
axes[1].legend(frameon=False)
for bar, rate in zip(bars, inv_rate.values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=12
    )

fig.suptitle(
    "Figure 5: Parental Involvement and Dropout Risk\n"
    "(Source: Ojo & Awonuga, 2022 — absent parents linked to 44% dropout rate)",
    fontsize=12, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "05_dropout_by_parental_involvement.png",
           "Dropout rate by parental involvement")


# ══════════════════════════════════════════════════════════════════════
# CHART 6 — Dropout Rate by Parental Education Level
# ══════════════════════════════════════════════════════════════════════
edu_order = [
    "No Formal Education", "Primary", "Secondary",
    "OND/NCE", "HND/BSc", "Postgraduate"
]
edu_labels = [
    "No Formal\nEdu", "Primary", "Secondary",
    "OND/NCE", "HND/BSc", "Postgrad"
]

fig, ax = plt.subplots(figsize=(11, 5))

edu_rate = (
    df.groupby("Parental_Education_Level")["Dropout_Status"]
    .mean() * 100
).reindex(edu_order)

edu_colors = [
    RED, "#C0392B", AMBER, "#2980B9", TEAL, "#1A5276"
]
bars = ax.bar(
    edu_labels, edu_rate.values,
    color=edu_colors, edgecolor=WHITE, linewidth=0.8, width=0.6
)
ax.set_title(
    "Figure 6: Parental Education Level and Dropout Rate\n"
    "(Source: World Bank GHS-Panel Nigeria)"
)
ax.set_xlabel("Parental Education Level")
ax.set_ylabel("Dropout Rate (%)")
ax.set_ylim(0, 65)
ax.axhline(
    df["Dropout_Status"].mean() * 100,
    color=PURPLE, linestyle="--", linewidth=1.5, label="Overall average (30.0%)"
)
ax.legend(frameon=False)
for bar, rate in zip(bars, edu_rate.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=10
    )
fig.tight_layout()
save_chart(fig, "06_dropout_by_parental_education.png",
           "Dropout rate by parental education level")


# ══════════════════════════════════════════════════════════════════════
# CHART 7 — Attendance Rate Distribution
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: overall histogram
axes[0].hist(
    df["Attendance_Rate_Pct"], bins=30,
    color=PURPLE, edgecolor=WHITE, linewidth=0.5, alpha=0.85
)
axes[0].axvline(
    df["Attendance_Rate_Pct"].mean(),
    color=AMBER, linewidth=2, linestyle="--",
    label=f"Mean: {df['Attendance_Rate_Pct'].mean():.1f}%"
)
axes[0].axvline(
    70, color=RED, linewidth=2, linestyle=":",
    label="Risk threshold: 70%"
)
axes[0].set_title("Overall Attendance Rate Distribution")
axes[0].set_xlabel("Attendance Rate (%)")
axes[0].set_ylabel("Number of Students")
axes[0].legend(frameon=False)

# Right: by dropout status
for status, color, label in zip(
    [0, 1], [TEAL, RED], ["Completed", "Dropped Out"]
):
    subset = df[df["Dropout_Status"] == status]["Attendance_Rate_Pct"]
    axes[1].hist(
        subset, bins=25, alpha=0.65,
        color=color, edgecolor=WHITE, linewidth=0.5, label=label
    )
axes[1].set_title("Attendance Rate by Dropout Status")
axes[1].set_xlabel("Attendance Rate (%)")
axes[1].set_ylabel("Number of Students")
axes[1].legend(frameon=False)

fig.suptitle(
    "Figure 7: Attendance Rate Distribution\n"
    "(Source: Hussain et al., 2023 — attendance strongest individual predictor, AUC contribution = 0.31)",
    fontsize=12, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "07_attendance_distribution.png",
           "Attendance rate distribution")


# ══════════════════════════════════════════════════════════════════════
# CHART 8 — Average Academic Score Distribution
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(
    df["Avg_Academic_Score_Pct"], bins=30,
    color=TEAL, edgecolor=WHITE, linewidth=0.5, alpha=0.85
)
axes[0].axvline(
    df["Avg_Academic_Score_Pct"].mean(),
    color=AMBER, linewidth=2, linestyle="--",
    label=f"Mean: {df['Avg_Academic_Score_Pct'].mean():.1f}%"
)
axes[0].axvline(
    40, color=RED, linewidth=2, linestyle=":",
    label="Fail threshold: 40%"
)
axes[0].set_title("Overall Academic Score Distribution")
axes[0].set_xlabel("Average Academic Score (%)")
axes[0].set_ylabel("Number of Students")
axes[0].legend(frameon=False)

for status, color, label in zip(
    [0, 1], [TEAL, RED], ["Completed", "Dropped Out"]
):
    subset = df[df["Dropout_Status"] == status]["Avg_Academic_Score_Pct"]
    axes[1].hist(
        subset, bins=25, alpha=0.65,
        color=color, edgecolor=WHITE, linewidth=0.5, label=label
    )
axes[1].set_title("Academic Score by Dropout Status")
axes[1].set_xlabel("Average Academic Score (%)")
axes[1].set_ylabel("Number of Students")
axes[1].legend(frameon=False)

fig.suptitle(
    "Figure 8: Average Academic Score Distribution\n"
    "(Source: Aluko et al., 2024 — Nigerian school mean ~40–45%)",
    fontsize=12, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "08_academic_score_distribution.png",
           "Academic score distribution")


# ══════════════════════════════════════════════════════════════════════
# CHART 9 — Cognitive Features: Learning Pace & Cognitive Score
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Learning pace dropout rate
pace_order = ["Slow", "Average", "Fast"]
pace_rate = (
    df.groupby("Learning_Pace")["Dropout_Status"]
    .mean() * 100
).reindex(pace_order)

bar_colors = [RED, AMBER, TEAL]
bars = axes[0].bar(
    pace_order, pace_rate.values,
    color=bar_colors, edgecolor=WHITE, linewidth=0.8, width=0.5
)
axes[0].set_title("Dropout Rate by Learning Pace")
axes[0].set_xlabel("Learning Pace (Teacher-Assessed)")
axes[0].set_ylabel("Dropout Rate (%)")
axes[0].set_ylim(0, 55)
axes[0].axhline(
    df["Dropout_Status"].mean() * 100,
    color=PURPLE, linestyle="--", linewidth=1.5
)
for bar, rate in zip(bars, pace_rate.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=11
    )

# Middle: Cognitive score histogram by dropout
for status, color, label in zip(
    [0, 1], [TEAL, RED], ["Completed", "Dropped Out"]
):
    subset = df[df["Dropout_Status"] == status]["Cognitive_Assessment_Score"]
    axes[1].hist(
        subset, bins=25, alpha=0.65,
        color=color, edgecolor=WHITE, linewidth=0.5, label=label
    )
axes[1].set_title("Cognitive Assessment Score\nby Dropout Status")
axes[1].set_xlabel("Cognitive Assessment Score")
axes[1].set_ylabel("Number of Students")
axes[1].legend(frameon=False)

# Right: Study habit score by dropout
for status, color, label in zip(
    [0, 1], [TEAL, RED], ["Completed", "Dropped Out"]
):
    subset = df[df["Dropout_Status"] == status]["Study_Habit_Score"]
    axes[2].hist(
        subset, bins=25, alpha=0.65,
        color=color, edgecolor=WHITE, linewidth=0.5, label=label
    )
axes[2].set_title("Study Habit Score\nby Dropout Status")
axes[2].set_xlabel("Study Habit Score")
axes[2].set_ylabel("Number of Students")
axes[2].legend(frameon=False)

fig.suptitle(
    "Figure 9: Cognitive Domain Features and Dropout Risk\n"
    "(Source: Dasi & Kanakala, 2022; FME, 2020; Zimmerman, 2002)",
    fontsize=12, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "09_cognitive_features.png",
           "Cognitive features and dropout risk")


# ══════════════════════════════════════════════════════════════════════
# CHART 10 — Psychomotor Features: PE & Psychomotor Score
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: PE participation dropout rate
pe_order = ["Absent", "Passive", "Active"]
pe_rate = (
    df.groupby("PE_Participation")["Dropout_Status"]
    .mean() * 100
).reindex(pe_order)

bar_colors = [RED, AMBER, TEAL]
bars = axes[0].bar(
    pe_order, pe_rate.values,
    color=bar_colors, edgecolor=WHITE, linewidth=0.8, width=0.5
)
axes[0].set_title("Dropout Rate by PE Participation")
axes[0].set_xlabel("PE Participation (Teacher-Assessed)")
axes[0].set_ylabel("Dropout Rate (%)")
axes[0].set_ylim(0, 60)
axes[0].axhline(
    df["Dropout_Status"].mean() * 100,
    color=PURPLE, linestyle="--", linewidth=1.5
)
for bar, rate in zip(bars, pe_rate.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=11
    )

# Middle: Practical subject score by dropout
for status, color, label in zip(
    [0, 1], [TEAL, RED], ["Completed", "Dropped Out"]
):
    subset = df[df["Dropout_Status"] == status]["Practical_Subject_Score"]
    axes[1].hist(
        subset, bins=25, alpha=0.65,
        color=color, edgecolor=WHITE, linewidth=0.5, label=label
    )
axes[1].set_title("Practical Subject Score\nby Dropout Status")
axes[1].set_xlabel("Practical Subject Score")
axes[1].set_ylabel("Number of Students")
axes[1].legend(frameon=False)

# Right: Psychomotor development score by dropout
for status, color, label in zip(
    [0, 1], [TEAL, RED], ["Completed", "Dropped Out"]
):
    subset = df[df["Dropout_Status"] == status]["Psychomotor_Dev_Score"]
    axes[2].hist(
        subset, bins=25, alpha=0.65,
        color=color, edgecolor=WHITE, linewidth=0.5, label=label
    )
axes[2].set_title("Psychomotor Development Score\nby Dropout Status")
axes[2].set_xlabel("Psychomotor Development Score")
axes[2].set_ylabel("Number of Students")
axes[2].legend(frameon=False)

fig.suptitle(
    "Figure 10: Psychomotor Domain Features and Dropout Risk\n"
    "(Source: Kizito, 2023; FME, 2020 — school belonging protective factor)",
    fontsize=12, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()
save_chart(fig, "10_psychomotor_features.png",
           "Psychomotor features and dropout risk")


# ══════════════════════════════════════════════════════════════════════
# CHART 11 — Cognitive Score vs Academic Score Scatter
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))

sample = df.sample(1500, random_state=42)

for status, color, label, alpha in zip(
    [0, 1], [TEAL, RED],
    ["Completed Year", "Dropped Out"], [0.4, 0.6]
):
    sub = sample[sample["Dropout_Status"] == status]
    ax.scatter(
        sub["Avg_Academic_Score_Pct"],
        sub["Cognitive_Assessment_Score"],
        c=color, alpha=alpha, s=18,
        label=f"{label} (n={len(sub):,})"
    )

ax.set_xlabel("Average Academic Score (%)")
ax.set_ylabel("Cognitive Assessment Score")
ax.set_title(
    "Figure 11: Cognitive Assessment Score vs Average Academic Score\n"
    "(1,500 random sample — showing moderate correlation, r ≈ 0.65)"
)
ax.legend(frameon=True, framealpha=0.9)

# Add correlation annotation
corr = df["Cognitive_Assessment_Score"].corr(df["Avg_Academic_Score_Pct"])
ax.annotate(
    f"Pearson r = {corr:.3f}",
    xy=(0.05, 0.93), xycoords="axes fraction",
    fontsize=11, color=PURPLE,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
              edgecolor=PURPLE, alpha=0.8)
)
fig.tight_layout()
save_chart(fig, "11_cognitive_vs_academic_scatter.png",
           "Cognitive vs academic score scatter")


# ══════════════════════════════════════════════════════════════════════
# CHART 12 — Dropout Rate by Distance to School (Binned)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bin distance
bins   = [0, 2, 5, 10, 15, 20, 35]
labels = ["0–2km", "2–5km", "5–10km", "10–15km", "15–20km", "20km+"]
df["Distance_Band"] = pd.cut(
    df["Distance_to_School_km"], bins=bins, labels=labels
)

# Left: distribution of students by distance band
dist_counts = df["Distance_Band"].value_counts().reindex(labels)
axes[0].bar(
    labels, dist_counts.values,
    color=PURPLE, edgecolor=WHITE, linewidth=0.8, alpha=0.85
)
axes[0].set_title("Student Distribution by Distance Band")
axes[0].set_xlabel("Distance from School")
axes[0].set_ylabel("Number of Students")
axes[0].set_xticklabels(labels, rotation=20, ha="right")

# Right: dropout rate by distance band
dist_rate = (
    df.groupby("Distance_Band", observed=True)["Dropout_Status"]
    .mean() * 100
).reindex(labels)

band_colors = [TEAL, TEAL, AMBER, AMBER, RED, RED]
bars = axes[1].bar(
    labels, dist_rate.values,
    color=band_colors, edgecolor=WHITE, linewidth=0.8
)
axes[1].set_title("Dropout Rate (%) by Distance Band")
axes[1].set_xlabel("Distance from School")
axes[1].set_ylabel("Dropout Rate (%)")
axes[1].set_ylim(0, 60)
axes[1].set_xticklabels(labels, rotation=20, ha="right")
axes[1].axhline(
    df["Dropout_Status"].mean() * 100,
    color=PURPLE, linestyle="--", linewidth=1.5, label="Overall average"
)
axes[1].axvline(
    2.5, color=RED, linestyle=":", linewidth=1.5, alpha=0.5,
    label="Risk threshold ~5km"
)
axes[1].legend(frameon=False)
for bar, rate in zip(bars, dist_rate.values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=9
    )

fig.suptitle(
    "Figure 12: Distance to School and Dropout Risk\n"
    "(Source: Fayomi & Nwosu, 2022 — risk rises sharply beyond 5km)",
    fontsize=12, fontweight="bold", color=PURPLE, y=1.02
)
fig.tight_layout()

# Drop temp column
df = df.drop(columns=["Distance_Band"])
save_chart(fig, "12_dropout_by_distance.png",
           "Dropout rate by distance band")


# ══════════════════════════════════════════════════════════════════════
# CHART 13 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 10))

# Select only numeric columns for correlation
numeric_cols = [
    "Age", "Distance_to_School_km",
    "Attendance_Rate_Pct", "Avg_Academic_Score_Pct",
    "Failed_Subjects_Count", "Cognitive_Assessment_Score",
    "Study_Habit_Score", "Practical_Subject_Score",
    "Psychomotor_Dev_Score", "Dropout_Status"
]

corr_matrix = df[numeric_cols].corr()

# Rename for display
display_names = {
    "Age":                       "Age",
    "Distance_to_School_km":     "Distance (km)",
    "Attendance_Rate_Pct":       "Attendance (%)",
    "Avg_Academic_Score_Pct":    "Academic Score (%)",
    "Failed_Subjects_Count":     "Failed Subjects",
    "Cognitive_Assessment_Score":"Cognitive Score",
    "Study_Habit_Score":         "Study Habits",
    "Practical_Subject_Score":   "Practical Score",
    "Psychomotor_Dev_Score":     "Psychomotor Score",
    "Dropout_Status":            "DROPOUT STATUS",
}
corr_matrix = corr_matrix.rename(
    index=display_names, columns=display_names
)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True, fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    linecolor=LIGHT_GRAY,
    cbar_kws={"shrink": 0.8},
    ax=ax,
    annot_kws={"size": 9}
)
ax.set_title(
    "Figure 13: Correlation Heatmap — Numeric Features\n"
    "(Values closer to -1 or +1 indicate stronger linear relationship with Dropout Status)",
    pad=20
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

fig.tight_layout()
save_chart(fig, "13_correlation_heatmap.png",
           "Correlation heatmap of numeric features")


# ══════════════════════════════════════════════════════════════════════
# CHART 14 — Summary: Dropout Rate Across All Categorical Features
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 8))

summary_data = []

categorical_features = {
    "Gender":                  {"Female": 1, "Male": 0},
    "Household_Income_Band":   None,
    "Fee_Payment_Status":      None,
    "Parental_Involvement":    None,
    "Learning_Pace":           None,
    "PE_Participation":        None,
    "Year_Group":              None,
}

for feat in categorical_features:
    rates = df.groupby(feat)["Dropout_Status"].mean() * 100
    for val, rate in rates.items():
        summary_data.append({
            "Label":      f"{feat.replace('_',' ')}: {val}",
            "Feature":    feat,
            "Value":      val,
            "Dropout_Rate": rate,
        })

summary_df = pd.DataFrame(summary_data).sort_values(
    "Dropout_Rate", ascending=True
)

bar_colors = [
    RED if r >= 45 else AMBER if r >= 30 else TEAL
    for r in summary_df["Dropout_Rate"]
]

bars = ax.barh(
    summary_df["Label"],
    summary_df["Dropout_Rate"],
    color=bar_colors, edgecolor=WHITE, linewidth=0.6, height=0.7
)
ax.axvline(
    df["Dropout_Status"].mean() * 100,
    color=PURPLE, linestyle="--", linewidth=2,
    label=f"Overall average: {df['Dropout_Status'].mean()*100:.1f}%"
)

for bar, rate in zip(bars, summary_df["Dropout_Rate"]):
    ax.text(
        rate + 0.5, bar.get_y() + bar.get_height() / 2,
        f"{rate:.1f}%", va="center", fontsize=8.5
    )

legend_patches = [
    mpatches.Patch(color=RED,   label="High risk (≥45%)"),
    mpatches.Patch(color=AMBER, label="Medium risk (30–44%)"),
    mpatches.Patch(color=TEAL,  label="Low risk (<30%)"),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([0], [0], color=PURPLE, linestyle="--",
               label=f"Overall average: {df['Dropout_Status'].mean()*100:.1f}%")
], frameon=False, loc="lower right")

ax.set_title(
    "Figure 14: Dropout Rate Across All Categorical Features\n"
    "(Summary comparison — all groups sorted by risk level)",
    pad=15
)
ax.set_xlabel("Dropout Rate (%)")
ax.set_xlim(0, 80)
fig.tight_layout()
save_chart(fig, "14_dropout_rate_all_categories.png",
           "Dropout rate summary across all categorical features")


# ══════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"  VISUALISATION COMPLETE")
print(f"{'=' * 65}")
print(f"  Total charts generated: {chart_count}")
print(f"  Saved to:  {CHART_DIR}/")
print(f"  Format:    PNG, 300 DPI (thesis-ready)")
print()
print("  Chart index:")
print("  01  Overall dropout distribution (pie)")
print("  02  Dropout by gender")
print("  03  Dropout by household income band")
print("  04  Dropout by fee payment status")
print("  05  Dropout by parental involvement")
print("  06  Dropout by parental education level")
print("  07  Attendance rate distribution")
print("  08  Academic score distribution")
print("  09  Cognitive domain features")
print("  10  Psychomotor domain features")
print("  11  Cognitive vs academic score scatter")
print("  12  Dropout by distance to school")
print("  13  Correlation heatmap")
print("  14  Dropout rate summary (all categories)")
print(f"\n  All charts are 300 DPI and ready for thesis insertion.")
print(f"{'=' * 65}\n")