import json
import csv

import pandas as pd

df_hc = pd.read_csv("hc_results/attack_stats_hc.csv")
df_hc_eps03 = df_hc[df_hc["hc_epsilon"] == 0.3]
df_wb = pd.read_csv("attack_results/attack_stats.csv")
df_wb = df_wb.rename(columns={"image": "image_file"})


df_hc_anneal = pd.read_csv("hc_results_annealing/attack_stats_hc_annealing.csv")
df_hc_anneal_eps03 = df_hc_anneal[df_hc["hc_epsilon"] == 0.3]
df_hc_anneal_pol = pd.read_csv("hc_results_annealing_polynomial/attack_stats_hc_annealing.csv")
df_hc_anneal_pol_eps03 = df_hc_anneal_pol[df_hc["hc_epsilon"] == 0.3]
df_hc_mut = pd.read_csv("hc_results_mutations/attack_stats_hc_more_mutations.csv")
df_hc_mut_eps03 = df_hc_mut[df_hc["hc_epsilon"] == 0.3]



# print(df_hc.columns)
# print(df_wb.columns)


# ============================================================
# Summarization Methods
# ============================================================
def print_attack_success_summary(df_hc, df_wb):
    print("=" * 60)
    print("ATTACK SUCCESS SUMMARY")
    print("=" * 60)

    # Hill Climber
    hc_success_rate = df_hc["hc_success"].mean()
    hc_changed_rate = df_hc["hc_changed_pred"].mean()

    print("\n[Hill Climber]")
    print(f"Success rate (hc_success):      {hc_success_rate:.3f}")
    print(f"Changed prediction rate:       {hc_changed_rate:.3f}")
    print(f"Total samples:                 {len(df_hc)}")

    # White-box
    for method in ["fgm", "pgd"]:
        success_rate = df_wb[f"{method}_success"].mean()
        changed_rate = df_wb[f"{method}_changed_pred"].mean()

        print(f"\n[{method.upper()}]")
        print(f"Success rate:                  {success_rate:.3f}")
        print(f"Changed prediction rate:       {changed_rate:.3f}")

    print("=" * 60)


def print_perturbation_summary(df_hc, df_wb):
    print("=" * 60)
    print("PERTURBATION CHARACTERISTICS")
    print("=" * 60)

    def summarize(df, prefix, name):
        print(f"\n[{name}]")
        print(f"Avg #pixels changed:  {df[f'{prefix}_num_pixels_changed'].mean():.2f}")
        print(f"Median pixels:        {df[f'{prefix}_num_pixels_changed'].median():.2f}")
        print(f"Max L_inf:            {df[f'{prefix}_linf'].max():.4f}")
        print(f"Mean L_inf:           {df[f'{prefix}_linf'].mean():.4f}")
        print(f"Mean L2:              {df[f'{prefix}_l2'].mean():.4f}")

    summarize(df_hc, "hc", "Hill Climber")
    summarize(df_wb, "fgm", "FGM")
    summarize(df_wb, "pgd", "PGD")

    print("=" * 60)


def print_runtime_summary(df_hc, df_wb):
    print("=" * 60)
    print("RUNTIME / EFFICIENCY")
    print("=" * 60)

    print("\n[Hill Climber]")
    print(f"Mean runtime (s): {df_hc['hc_runtime_s'].mean():.3f}")
    print(f"Median runtime:   {df_hc['hc_runtime_s'].median():.3f}")
    print(f"Max runtime:      {df_hc['hc_runtime_s'].max():.3f}")

    for method in ["fgm", "pgd"]:
        print(f"\n[{method.upper()}]")
        print(f"Mean runtime (s): {df_wb[f'{method}_runtime_s'].mean():.3f}")
        print(f"Median runtime:   {df_wb[f'{method}_runtime_s'].median():.3f}")
        print(f"Max runtime:      {df_wb[f'{method}_runtime_s'].max():.3f}")

    print("=" * 60)









# ============================================================
# Report Plotting Methods
# ============================================================
def plot_success_rates(df_hc, df_wb):
    import matplotlib.pyplot as plt

    methods = ["FGM", "PGD", "HC"]
    rates = [
        df_wb["fgm_success"].mean(),
        df_wb["pgd_success"].mean(),
        df_hc["hc_success"].mean()
    ]

    plt.figure()
    plt.bar(methods, rates)
    plt.ylabel("Success rate")
    plt.ylim(0,1)
    plt.title("Attack Success Rate")
    plt.show()

def plot_perturbation_tradeoff(df_hc, df_wb):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(df_wb["fgm_num_pixels_changed"], df_wb["fgm_linf"], label="FGM", alpha=0.6)
    plt.scatter(df_wb["pgd_num_pixels_changed"], df_wb["pgd_linf"], label="PGD", alpha=0.6)
    plt.scatter(df_hc["hc_num_pixels_changed"], df_hc["hc_linf"], label="HC", alpha=0.6)

    plt.xlabel("# Pixels changed")
    plt.ylabel("L∞")
    plt.legend()
    plt.title("Perturbation Characteristics")
    plt.show()

def plot_runtime_vs_success(df_hc, df_wb):
    import matplotlib.pyplot as plt

    runtimes = [
        df_wb["fgm_runtime_s"].mean(),
        df_wb["pgd_runtime_s"].mean(),
        df_hc["hc_runtime_s"].mean()
    ]
    success = [
        df_wb["fgm_success"].mean(),
        df_wb["pgd_success"].mean(),
        df_hc["hc_success"].mean()
    ]

    labels = ["FGM", "PGD", "HC"]

    plt.figure()
    plt.scatter(runtimes, success)

    for i, lbl in enumerate(labels):
        plt.annotate(lbl, (runtimes[i], success[i]))

    plt.xlabel("Mean runtime (s)")
    plt.ylabel("Success rate")
    plt.title("Efficiency vs Effectiveness")
    plt.show()


# print_attack_success_summary(df_hc_anneal_eps03, df_wb)
# print_perturbation_summary(df_hc_anneal_eps03, df_wb)
# print_runtime_summary(df_hc_anneal_eps03, df_wb)
#
# print_attack_success_summary(df_hc_anneal_pol_eps03, df_wb)
# print_perturbation_summary(df_hc_anneal_pol_eps03, df_wb)
# print_runtime_summary(df_hc_anneal_pol_eps03, df_wb)

print_attack_success_summary(df_hc_mut_eps03, df_wb)
print_perturbation_summary(df_hc_mut_eps03, df_wb)
print_runtime_summary(df_hc_mut_eps03, df_wb)
#
# plot_success_rates(df_hc_eps03, df_wb)
# plot_perturbation_tradeoff(df_hc_eps03, df_wb)
# plot_runtime_vs_success(df_hc_eps03, df_wb)

#
# hc_eps_stats = (
#     df_hc
#     .groupby("hc_epsilon")
#     .agg(
#         avg_runtime_s=("hc_runtime_s", "mean"),
#         median_runtime_s=("hc_runtime_s", "median"),
#         avg_pixels_changed=("hc_num_pixels_changed", "mean"),
#         median_pixels_changed=("hc_num_pixels_changed", "median"),
#         success_rate=("hc_success", "mean"),
#         n_samples=("hc_success", "count")
#     )
#     .reset_index()
# )
# print("=" * 70)
# print("HILL CLIMBER STATISTICS PER EPSILON")
# print("=" * 70)
#
# for _, r in hc_eps_stats.iterrows():
#     print(f"\nε = {r.hc_epsilon:.3f}")
#     print(f"  Samples:              {int(r.n_samples)}")
#     print(f"  Success rate:         {r.success_rate:.3f}")
#     print(f"  Avg runtime (s):      {r.avg_runtime_s:.2f}")
#     print(f"  Median runtime (s):   {r.median_runtime_s:.2f}")
#     print(f"  Avg pixels changed:   {r.avg_pixels_changed:.1f}")
#     print(f"  Median pixels:        {r.median_pixels_changed:.1f}")
#
# print("=" * 70)




