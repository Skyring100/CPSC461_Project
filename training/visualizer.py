import matplotlib.pyplot as plt

from utils import ensure_parent_dir


def save_comparison_plot(training_results: dict, version: str, save_path: str, title: str):
    """Saves a comparison figure for CNN vs ConvKAN on a MedMNIST version"""
    if not ("cnn" in training_results and "convkan" in training_results):
        return

    models_list = ["cnn", "convkan"]
    stats = training_results
    colors = ["#3498db", "#e74c3c"]

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    # Row 1 – bar charts
    _bar_chart(fig.add_subplot(gs[0, 0]), models_list,
               [stats[m]["num_params"] / 1e6 for m in models_list],
               colors, "Parameters (Millions)", "Total Parameters",
               fmt="{:.2f}M")

    _bar_chart(fig.add_subplot(gs[0, 1]), models_list,
               [stats[m]["accuracy"] for m in models_list],
               colors, "Accuracy (%)", "Overall Accuracy",
               fmt="{:.2f}%", ylim=(0, 105))

    _bar_chart(fig.add_subplot(gs[0, 2]), models_list,
               [stats[m]["f1"] for m in models_list],
               colors, "F1 Score", "F1 Score (Weighted)",
               fmt="{:.4f}", ylim=(0, 1.1))

    _bar_chart(fig.add_subplot(gs[0, 3]), models_list,
               [stats[m]["total_time"] / 60 for m in models_list],
               colors, "Time (minutes)", "Total Training Time",
               fmt="{:.1f}m")

    # Row 2 – metrics table
    _metrics_table(fig.add_subplot(gs[1, :]), stats)

    # Row 3 – confusion matrices + training curves
    for idx, model_name in enumerate(models_list):
        _confusion_matrix_plot(fig.add_subplot(gs[2, idx]),
                               stats[model_name]["confusion_matrix"],
                               model_name)

    _line_plot(fig.add_subplot(gs[2, 2]), models_list, colors,
               key="train_loss", stats=stats,
               xlabel="Epoch", ylabel="Loss", title="Loss Curves",
               label_fmt="{} Loss")

    _line_plot(fig.add_subplot(gs[2, 3]), models_list, colors,
               key="val_acc", stats=stats,
               xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy Curves",
               label_fmt="{} Acc")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Private helpers

def _bar_chart(ax, models_list, values, colors, ylabel, title, fmt, ylim=None):
    bars = ax.bar(models_list, values, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    if ylim:
        ax.set_ylim(ylim)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                fmt.format(val), ha="center", va="bottom",
                fontsize=10, fontweight="bold")


def _metrics_table(ax, stats):
    ax.axis("off")
    cnn, kan = stats["cnn"], stats["convkan"]
    headers = ["Metric", "CNN", "ConvKAN"]
    rows = [
        ["Total Parameters",      f"{cnn['num_params']:,}",               f"{kan['num_params']:,}"],
        ["Overall Accuracy",      f"{cnn['accuracy']:.2f}%",              f"{kan['accuracy']:.2f}%"],
        ["Precision (Weighted)",  f"{cnn['precision']:.4f}",              f"{kan['precision']:.4f}"],
        ["Recall (Weighted)",     f"{cnn['recall']:.4f}",                 f"{kan['recall']:.4f}"],
        ["F1 Score (Weighted)",   f"{cnn['f1']:.4f}",                     f"{kan['f1']:.4f}"],
        ["Peak Training RAM",     f"{cnn['peak_ram']:.2f} GB",            f"{kan['peak_ram']:.2f} GB"],
        ["Peak Training VRAM",    f"{cnn['peak_vram']:.2f} GB",           f"{kan['peak_vram']:.2f} GB"],
        ["Total Training Time",   f"{cnn['total_time']/60:.2f} min",      f"{kan['total_time']/60:.2f} min"],
        ["Avg Time per Epoch",    f"{cnn['avg_epoch_time']:.2f}s",        f"{kan['avg_epoch_time']:.2f}s"],
        ["Total Epochs",          str(cnn["num_epochs"]),                  str(kan["num_epochs"])],
    ]

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                     loc="center", colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style the table
    for col in range(len(headers)):
        table[(0, col)].set_facecolor("#34495e")
        table[(0, col)].set_text_props(weight="bold", color="white")

    # Style the rows
    for row in range(1, len(rows) + 1):
        for col in range(len(headers)):
            table[(row, col)].set_facecolor("#ecf0f1" if row % 2 == 0 else "#ffffff")


def _confusion_matrix_plot(ax, cm, model_name: str):
    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"], fontsize=10)
    ax.set_yticklabels(["Negative", "Positive"], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
    ax.set_ylabel("True", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix – {model_name.upper()}", fontsize=12, fontweight="bold")

    # Add the values to the confusion matrix
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")


def _line_plot(ax, models_list, colors, key, stats, xlabel, ylabel, title, label_fmt):
    for idx, model_name in enumerate(models_list):
        ax.plot(stats[model_name][key],
                label=label_fmt.format(model_name.upper()),
                color=colors[idx], linewidth=2, marker="o", markersize=3)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)