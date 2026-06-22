# LPA GPU Optimization Attempts

## Context

The baseline (`tnl_label_propagation.cu`) runs LPA with `parallelFor<Cuda>(0, n)` — one GPU thread per node, each thread iterates its neighbours sequentially and picks the most frequent label. All `n` nodes are processed every iteration regardless of how many actually changed.

---

## 1. Frontier-based optimization — `tnl_lpa_frontier.cu`

### Idea
A node's label can only change if at least one neighbour changed last round. So instead of processing all `n` nodes every iteration, maintain a compact **frontier** — the set of neighbours of nodes that changed — and only process those.

### Implementation
After each iteration:
1. For each changed node, mark all its neighbours as `active_next` (push phase).
2. Count active nodes via `TNL::Algorithms::reduce`.
3. Copy `active_next` to `scan_buf`, run `TNL::Algorithms::inplaceExclusiveScan` to get scatter positions.
4. Scatter active indices into a compact `frontier` array via `parallelFor`.
5. Next iteration: `parallelFor(0, frontier_size)` instead of `parallelFor(0, n)`.

### Why it failed
The frontier management itself is **O(n) every iteration** regardless of how small the frontier is:
- `active_next` reset (GPU memset over n elements)
- `scan_buf = active_next` (O(n) copy)
- `inplaceExclusiveScan` (O(n) prefix sum)
- scatter `parallelFor(0, n)` (O(n))

LPA also converges quickly — the first several iterations have frontier ≈ n anyway, so the overhead dominates the entire run. Net result: slightly slower than baseline.

### What would help instead
A bitmask-only approach (no compaction): keep `active[n]`, skip inactive nodes with an early `return` inside the kernel, zero only the frontier nodes each iteration. Avoids the scan/scatter/copy cost while still skipping inactive nodes.

---

## 2. SlicedEllpack matrix format — first attempt in `tnl_lpa_ellpack.cu`

### Idea
The baseline uses CSR, which stores rows contiguously. When 32 GPU threads (one warp) each access element `i` of their respective rows, the memory locations are scattered → not coalesced. **SlicedEllpack** groups 32 rows into a slice and stores element `i` of all 32 rows contiguously, so warp threads accessing the same element index land in one cache line → coalesced reads.

### Implementation
Change the `Segments` template parameter of `SparseMatrix`:
```cpp
using DeviceMatrix = TNL::Matrices::SparseMatrix<
    RealType, TNL::Devices::Cuda, IndexType,
    TNL::Matrices::GeneralMatrix,
    TNL::Algorithms::Segments::SlicedEllpack>;
```
Host-side loading stays CSR (`HostMatrix`) since `SlicedEllpack::setRowCapacities` calls `TNL::sum` which is not defined for Host arrays. TNL handles the format conversion on `device_matrix = host_matrix`.

### Issues encountered
- **Compile error**: `TNL::sum` not defined for Host arrays → solved by splitting into separate `HostMatrix` (CSR) and `DeviceMatrix` (SlicedEllpack) types.
- **Runtime crash**: padding elements in SlicedEllpack have column index `-1`; accessing `labels[-1]` triggered a bounds assertion → solved by adding `if (neighbor < 0) continue` in the kernel loop.

### Why it was slower
SlicedEllpack pads all rows in a 32-row slice to the **maximum degree** in that slice. Citeseer is a power-law graph — degree varies enormously. One hub node per slice forces all 31 other nodes (potentially degree 5) to iterate 800+ times, wasting GPU cycles on `neighbor < 0` checks. This overhead outweighs the coalescing benefit.

---

## 3. BiEllpack matrix format — second attempt in `tnl_lpa_ellpack.cu`

### Idea
Same as SlicedEllpack but **sorts rows within each slice by degree** before padding. Low-degree nodes are grouped together and high-degree nodes are grouped together, so the padding waste per slice is minimised. This is the SELL-C-σ idea from the literature.

### Implementation
One-line change from the SlicedEllpack version:
```cpp
TNL::Algorithms::Segments::BiEllpack   // instead of SlicedEllpack
```
Same template signature; same `neighbor < 0` guard needed for residual padding.

### Why it was slower
Even with sorting, BiEllpack adds overhead (sorting rows at construction time). The fundamental problem remains: LPA's inner loop is **sequential per thread** (maintaining a label frequency table), so the warp-level coalescing benefit of Ellpack formats is limited. The warp threads diverge anyway as soon as they start processing neighbours of different degrees, and the label lookup `labels[neighbour]` is random global memory access — coalescing the column-index reads helps but doesn't eliminate the cache-miss bottleneck on labels.

---

## 4. Things investigated but not implemented

### `forSegments` (TNL Segments API)
- **How it works**: iterates in parallel over segments (rows), but sequentially within each segment — i.e., one thread per row.
- **Why not useful**: identical thread assignment to the current `parallelFor + getRow` pattern; no load-balancing benefit for varying degree.

### `reduceSegments` (TNL Segments API)
- **How it works**: parallel reduction within each segment using the CSR Adaptive/Vector kernel (warp per row). Can assign 32 threads to one row.
- **Why not applicable**: requires a simple binary reduction (`a op b → result`). LPA's inner loop builds a label frequency table — this cannot be expressed as an associative binary operation.

### Register hash table for label counting
- **Idea**: replace the linear scan over `label_ids[]` (O(degree × 32)) with a small open-addressing hash table in registers (O(degree) average). Would help high-degree nodes.
- **Not implemented**: moderate complexity for uncertain gain given the memory-access bottleneck dominates.

---

## Root cause summary

LPA on GPU is fundamentally hard to accelerate beyond a basic implementation because:

1. **Random memory access**: `labels[neighbour]` is a random global memory read for every neighbour of every node — no access pattern helps with this.
2. **Sequential per-node logic**: building the label frequency table cannot be parallelised across threads for a single node without shared memory and complex coordination.
3. **Fast convergence**: LPA typically converges in 10–30 iterations, so the total GPU time is small. Optimisations that pay off per-iteration (frontier, format) add fixed overhead that often exceeds the savings.

The biggest realistic gain would come from **preprocessing the graph** (reordering nodes by BFS/degree to improve cache locality of `labels[neighbour]`) — this is an algorithmic change independent of TNL.
