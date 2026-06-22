import pandas as pd
import matplotlib.pyplot as plt

# ---- Load data ----
# File produced by run_scaling.sh
filename = "communities_cuda.txt"

df = pd.read_csv(
    filename,
    comment="#",
    sep=r"\s+",
    names=["nodes", "communities", "time"]
)
# -----------------------------
# Create figure
# -----------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))

# ---- Runtime (LEFT AXIS) ----
color_time = "tab:blue"

line1 = ax1.plot(
    df["nodes"],
    df["time"],
    marker="o",
    color=color_time,
    label="Runtime (seconds)"
)

ax1.set_xlabel("Number of Nodes")
ax1.set_ylabel("LPA Runtime (seconds)", color=color_time)
ax1.tick_params(axis="y", labelcolor=color_time)

ax1.grid(True, linestyle="--", alpha=0.6)

# ---- Communities (RIGHT AXIS) ----
ax2 = ax1.twinx()

color_comm = "tab:red"

line2 = ax2.plot(
    df["nodes"],
    df["communities"],
    marker="s",
    color=color_comm,
    label="Communities"
)

ax2.set_ylabel("Number of Communities", color=color_comm)
ax2.tick_params(axis="y", labelcolor=color_comm)

# -----------------------------
# Combined legend
# -----------------------------
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

# -----------------------------
# Title
# -----------------------------
plt.title("LPA Scaling: Runtime and Community Count vs Graph Size")
output_file = "communities_cuda.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")

print(f"Plot saved as: {output_file}")

# Optional: show window
plt.show()