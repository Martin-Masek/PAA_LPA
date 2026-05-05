import matplotlib.pyplot as plt

# Replace these with your actual file paths
cpu_file = "communities_cpu_16_citeseer_2.txt"
cuda_file = "communities_cuda_citeseer_2.txt"

def load_times(filename):
    times = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Assuming the time is the only value in the line or last column
            times.append(float(line.split()[-1]))
    return times

cpu_times = load_times(cpu_file)
cuda_times = load_times(cuda_file)

# Make sure both lists have the same length
n = min(len(cpu_times), len(cuda_times))
cpu_times = cpu_times[:n]
cuda_times = cuda_times[:n]

# Plot side-by-side bar chart
# ... (loading logic remains the same)

# 1. Define your step size
step = 50000

# 2. Generate x values as multiples of the step
x = [i * step for i in range(n)]

# 3. Adjust width to be relative to the step (e.g., 35% of the step size)
width = step * 0.35

fig, ax = plt.subplots(figsize=(10,6))

# 4. Use the new x and width values
ax.bar([val - width/2 for val in x], cpu_times, width=width, label='CPU 16 cores')
ax.bar([val + width/2 for val in x], cuda_times, width=width, label='CUDA')

ax.set_xlabel("Test / Graph index (Scale: 50,000)")
ax.set_ylabel("Time (s)")
ax.set_title("CPU vs CUDA execution times")
ax.legend()

# Optional: Format x-axis with commas for better readability
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.savefig("cpu_vs_cuda_times_citeseer_2.png", dpi=300)