import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame


def plot_parallel_consistent(sub_df, title, legend_labels):
    plt.figure(figsize=(10, 6))

    present_epsilons = sub_df["hc_epsilon"].cat.remove_unused_categories().unique()

    current_colors = [COLOR_MAP[eps] for eps in present_epsilons]

    ax = pd.plotting.parallel_coordinates(
        sub_df[[*cols_to_display, "hc_epsilon"]],
        'hc_epsilon',
        color=current_colors
    )

    handles, labels = ax.get_legend_handles_labels()

    new_labels = []
    for eps in labels:
        eps = float(eps)
        n = legend_labels.get(eps, 0)
        new_labels.append(f"{eps} (n={n})")

    ax.legend(handles, new_labels, title="hc_epsilon")

    plt.title(title)
    plt.savefig(os.path.join(CURRENT_DIR, FIG_DIRNAME, f"{HC_MODE}_parallel_{title.split(' ')[0].lower()}.png"))
    plt.show()


def parallel_coordinates_plots(df: DataFrame):
    df["hc_epsilon"] = df["hc_epsilon"].astype(float)

    df["hc_epsilon"] = pd.Categorical(df["hc_epsilon"], categories=ALL_EPSILONS, ordered=True)

    df['hc_runtime_s'] = (
            (df['hc_runtime_s'] - df['hc_runtime_s'].min()) /
            (df['hc_runtime_s'].max() - df['hc_runtime_s'].min())
    )

    df['hc_l2'] = (
            (df['hc_l2'] - df['hc_l2'].min()) /
            (df['hc_l2'].max() - df['hc_l2'].min())
    )

    df_changed_pred = df.query('hc_changed_pred == 1').copy()
    df_unchanged_pred = df.query('hc_changed_pred == 0').copy()

    epsilon_counts_changed = df_changed_pred["hc_epsilon"].value_counts().sort_index()
    epsilon_counts_unchanged = df_unchanged_pred["hc_epsilon"].value_counts().sort_index()

    plot_parallel_consistent(df_changed_pred, "Successful Attacks (Changed Pred)", epsilon_counts_changed)
    plot_parallel_consistent(df_unchanged_pred, "Failed Attacks (Unchanged Pred)", epsilon_counts_unchanged)

def bar_charts(df: DataFrame):
    unique_images = df["image_file"].unique()

    # split into groups of 5 images
    chunk_size = 5
    image_chunks = [unique_images[i:i + chunk_size] for i in range(0, len(unique_images), chunk_size)]

    from matplotlib.patches import Patch

    for plot_idx, group in enumerate(image_chunks):
        fig, ax = plt.subplots(figsize=(14, 7))

        x_indices = range(len(group))
        width = 0.2
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

        for img_i, img_name in enumerate(group):
            img_data = df[df["image_file"] == img_name]

            for eps_i, eps in enumerate(ALL_EPSILONS):
                row = img_data[img_data["hc_epsilon"] == float(eps)]

                if not row.empty:
                    runtime = row["hc_runtime_s"].values[0]
                    is_success = row["hc_changed_pred"].values[0] == 1

                    # COLOR LOGIC:
                    # If Success -> Use the Epsilon Color
                    # If Fail    -> Use Grey
                    bar_color = COLOR_MAP[eps] if is_success else "#D3D3D3"
                    edge_color = "none" if is_success else "#888888"

                    ax.bar(
                        x_indices[img_i] + offsets[eps_i],
                        runtime,
                        width,
                        color=bar_color,
                        edgecolor=edge_color,
                        zorder=3
                    )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(group, fontsize=11, fontweight='bold')
        ax.set_ylabel("Runtime (seconds)", fontsize=12)
        ax.set_title(f"Attack Efficiency per Image (Set {plot_idx + 1})", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

        legend_elements = [
            Patch(facecolor=COLOR_MAP[eps], label=f"Eps {eps} (Success)")
            for eps in ALL_EPSILONS
        ]
        legend_elements.append(Patch(facecolor="#D3D3D3", edgecolor="#888888", label="Failed Attack"))

        ax.legend(handles=legend_elements, title="Outcome & Epsilon", loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(CURRENT_DIR, FIG_DIRNAME, f"{HC_MODE}_barpart_{plot_idx + 1}.png"))
        plt.show()

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    CURRENT_DIR = os.path.dirname(__file__)
    ALL_EPSILONS = [0.05, 0.1, 0.2, 0.3]
    ALL_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2"]
    COLOR_MAP = dict(zip(ALL_EPSILONS, ALL_COLORS))
    FIG_DIRNAME = "eval_epsilon_plots"
    # HC_MODE = "hc_base"
    HC_MODE = "hc_extended"

    cols_to_display = [
        "hc_runtime_s",
        "hc_linf",
        "hc_l2",
        "hc_changed_perc"
    ]

    if not os.path.isdir(os.path.join(CURRENT_DIR, FIG_DIRNAME)):
        os.mkdir(os.path.join(CURRENT_DIR, FIG_DIRNAME))

    df = None

    if HC_MODE == "hc_base":
        df = pd.read_csv(os.path.join(CURRENT_DIR, "hc_results", "attack_stats_hc.csv"))
    elif HC_MODE == "hc_extended":
        df = pd.read_csv(os.path.join(CURRENT_DIR, "hc_results_annealing_mutations", "attack_stats.csv"))
    else:
        raise Exception("Unsupported HC mode")

    parallel_coordinates_plots(df.copy())

    bar_charts(df.copy())
