import pandas as pd
import matplotlib.pyplot as plt

# Load the results
filename = "cpu_thread_scaling.txt"

df = pd.read_csv(
    filename,
    comment="#",
    sep=r"\s+",
    names=["threads", "communities", "time"]
)

# Plot runtime vs threads
plt.figure(figsize=(7, 5))
plt.plot(df["threads"], df["time"], marker="o", color="tab:blue")

plt.xlabel("Number of Threads")
plt.ylabel("LPA Runtime (seconds)")
plt.title("CPU Thread Scaling for LPA (500,000 nodes)")
plt.grid(True, linestyle="--", alpha=0.6)

# Optional: mark points
for x, y in zip(df["threads"], df["time"]):
    plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

# Save figure
output_file = "cpu_thread_scaling.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved to {output_file}")

plt.show()