import sys
import pandas as pd
import matplotlib.pyplot as plt

filename = sys.argv[1] if len(sys.argv) > 1 else "bench_max_labels_citeseer.csv"

df = pd.read_csv(filename)

fig, ax1 = plt.subplots(figsize=(9, 5))

color_time = "tab:blue"
line1 = ax1.plot(
    df["max_labels"],
    df["time_s"],
    marker="o",
    color=color_time,
    label="Runtime (seconds)"
)
ax1.set_xlabel("MAX_LOCAL_LABELS")
ax1.set_ylabel("LPA Runtime (seconds)", color=color_time)
ax1.tick_params(axis="y", labelcolor=color_time)
ax1.set_xscale("log", base=2)
ax1.set_xticks(df["max_labels"])
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.grid(True, linestyle="--", alpha=0.6)

ax2 = ax1.twinx()
color_comm = "tab:red"
line2 = ax2.plot(
    df["max_labels"],
    df["communities"],
    marker="s",
    color=color_comm,
    label="Communities"
)
ax2.set_ylabel("Number of Communities", color=color_comm)
ax2.tick_params(axis="y", labelcolor=color_comm)

lines  = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

plt.title("LPA: Runtime and Communities vs MAX_LOCAL_LABELS")
output_file = filename.replace(".csv", ".png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved as: {output_file}")
plt.show()
