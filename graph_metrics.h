#pragma once
#include <iostream>
#include <iomanip>       // std::setw, std::setprecision, std::fixed
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <string>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>

// ── Type aliases (must be declared before use below) ─────────────────────────
using RealType  = float;
using IndexType = int;

template<typename Device>
using GraphMatrix = TNL::Matrices::SparseMatrix<RealType, Device, IndexType>;

// ─── GraphMetrics struct ──────────────────────────────────────────────────────
struct GraphMetrics {
    // --- Basic structure ---
    int       n_nodes;
    long long n_edges;        // undirected edge count
    double    avg_degree;
    int       max_degree;
    int       min_degree;
    double    degree_std_dev;

    // --- Degree distribution shape ---
    double gini_coefficient;  // 0 = perfectly uniform, 1 = one node has all edges

    // --- Memory access irregularity ---
    double label_working_set_mb;   // labels[] array size in MB.
                                   // If < CPU L3 cache size, CPU cache hides random access.
    double avg_neighbor_id_spread; // Avg std-dev of neighbor IDs per node.
                                   // Low  = good locality (BFS-ordered graph).
                                   // High = scattered = cache misses on every labels[] read.

    // --- GPU-specific signals ---

    // p10_degree / p90_degree  (10th and 90th percentile degrees).
    // Better than min/max because a handful of degree-0 isolated nodes
    // or one giant hub no longer destroys the metric.
    // Close to 1.0 → threads in a warp do similar work → low divergence.
    // Close to 0.0 → huge degree variance → most warp threads idle.
    double warp_efficiency_estimate;
    int    p10_degree;
    int    p90_degree;

    // LPA arithmetic intensity is always ~0.5 ops/byte regardless of graph.
    // Included for reference — confirms LPA is always memory-bound, never compute-bound.
    double arithmetic_intensity_estimate;

    // --- Prediction ---
    enum class Recommendation { CPU, CUDA, UNCLEAR };
    Recommendation recommendation;
    std::string    recommendation_reason;
};

// ─── compute_graph_metrics ────────────────────────────────────────────────────
// Pass a HOST-side matrix. If running on CUDA device, copy first:
//   GraphMatrix<TNL::Devices::Host> host_m; host_m = device_matrix;
//   auto gm = compute_graph_metrics(host_m);

GraphMetrics compute_graph_metrics(const GraphMatrix<TNL::Devices::Host>& m,
                                   bool verbose = true)
{
    GraphMetrics gm;
    int n = m.getRows();
    gm.n_nodes = n;

    std::vector<int>    degrees(n, 0);
    std::vector<double> neighbor_spreads(n, 0.0);
    long long total_edges = 0;

    // ── 1. Per-node degree + neighbor-ID spread ───────────────────────────
    for (int v = 0; v < n; v++) {
        auto row = m.getRow(v);
        std::vector<int> nbr_ids;

        for (int i = 0; i < row.getSize(); i++) {
            if (row.getValue(i) == 0.0f) continue;
            nbr_ids.push_back(row.getColumnIndex(i));
        }

        int deg = (int)nbr_ids.size();
        degrees[v]   = deg;
        total_edges += deg;

        if (deg >= 2) {
            double mean = 0;
            for (int id : nbr_ids) mean += id;
            mean /= deg;

            double var = 0;
            for (int id : nbr_ids) var += (id - mean) * (id - mean);
            neighbor_spreads[v] = std::sqrt(var / deg);
        }
    }

    gm.n_edges    = total_edges / 2;
    gm.max_degree = *std::max_element(degrees.begin(), degrees.end());
    gm.min_degree = *std::min_element(degrees.begin(), degrees.end());

    double sum_deg = std::accumulate(degrees.begin(), degrees.end(), 0.0);
    gm.avg_degree  = sum_deg / n;

    double var_deg = 0;
    for (int d : degrees) var_deg += (d - gm.avg_degree) * (d - gm.avg_degree);
    gm.degree_std_dev = std::sqrt(var_deg / n);

    // ── 2. Gini coefficient ───────────────────────────────────────────────
    {
        std::vector<int> sorted_deg = degrees;
        std::sort(sorted_deg.begin(), sorted_deg.end());
        double weighted_sum = 0;
        for (int i = 0; i < n; i++)
            weighted_sum += (double)(i + 1) * sorted_deg[i];
        gm.gini_coefficient = (2.0 * weighted_sum) / (n * sum_deg) - (double)(n + 1) / n;

        // Percentile-based warp efficiency (robust to outliers).
        // p10/p90 instead of min/max so a few zero-degree isolated nodes
        // or one extreme hub don't collapse the metric to ~0.
        gm.p10_degree = sorted_deg[(int)(0.10 * n)];
        gm.p90_degree = sorted_deg[(int)(0.90 * n)];
        gm.warp_efficiency_estimate = (gm.p90_degree > 0)
            ? (double)gm.p10_degree / gm.p90_degree
            : 1.0;
    }

    // ── 3. Memory metrics ─────────────────────────────────────────────────
    gm.label_working_set_mb = (n * 4.0) / (1024.0 * 1024.0);

    {
        double total_spread = 0; int counted = 0;
        for (int v = 0; v < n; v++) {
            if (degrees[v] >= 2) { total_spread += neighbor_spreads[v]; counted++; }
        }
        gm.avg_neighbor_id_spread = counted > 0 ? total_spread / counted : 0;
    }

    // ── 4. Arithmetic intensity ───────────────────────────────────────────
    // LPA: ~2 ops (add + compare) per neighbor, reads 4 bytes (one int label).
    // → always ~0.5 ops/byte, always memory-bound.
    gm.arithmetic_intensity_estimate = 0.5;

    // ── 5. CPU vs CUDA recommendation heuristic ───────────────────────────
    //
    // CPU wins:  small graph (labels[] fits in L3 cache) + sparse + uneven degrees
    // CUDA wins: large graph (labels[] >> L3) + dense or uniform degrees
    bool fits_in_cache   = gm.label_working_set_mb < 16.0;
    bool sparse          = gm.avg_degree < 20.0;
    bool uneven_degrees  = gm.gini_coefficient > 0.5;
    bool very_large      = gm.label_working_set_mb > 32.0;
    bool dense           = gm.avg_degree > 50.0;
    bool uniform_degrees = gm.gini_coefficient < 0.3;

    if (fits_in_cache && sparse && uneven_degrees) {
        gm.recommendation = GraphMetrics::Recommendation::CPU;
        gm.recommendation_reason =
            "labels[] fits in L3 cache (" +
            std::to_string((int)gm.label_working_set_mb) + " MB), sparse (avg_deg=" +
            std::to_string((int)gm.avg_degree) + "), uneven degrees (gini=" +
            std::to_string(gm.gini_coefficient).substr(0, 4) + ") -> poor GPU load balance.";
    } else if (very_large && (dense || uniform_degrees)) {
        gm.recommendation = GraphMetrics::Recommendation::CUDA;
        gm.recommendation_reason =
            "labels[] too large for CPU cache (" +
            std::to_string((int)gm.label_working_set_mb) + " MB), " +
            (dense ? "dense graph" : "uniform degree distribution") +
            " -> GPU bandwidth + parallelism wins.";
    } else {
        gm.recommendation = GraphMetrics::Recommendation::UNCLEAR;
        gm.recommendation_reason =
            "Mixed signals — benchmark both. labels[]=" +
            std::to_string((int)gm.label_working_set_mb) + " MB, avg_deg=" +
            std::to_string(gm.avg_degree).substr(0, 5) + ", gini=" +
            std::to_string(gm.gini_coefficient).substr(0, 4) + ".";
    }

    // ── 6. Print ──────────────────────────────────────────────────────────
    if (verbose) {
        std::cout << "\n╔══════════════════════════════════════════════════╗\n";
        std::cout <<   "║              GRAPH METRICS FOR LPA              ║\n";
        std::cout <<   "╚══════════════════════════════════════════════════╝\n";

        std::cout << "\n── Basic structure ─────────────────────────────────\n";
        std::cout << "  Nodes:              " << gm.n_nodes        << "\n";
        std::cout << "  Edges:              " << gm.n_edges        << "\n";
        std::cout << "  Avg degree:         " << gm.avg_degree     << "\n";
        std::cout << "  Max degree:         " << gm.max_degree     << "\n";
        std::cout << "  Min degree:         " << gm.min_degree     << "\n";
        std::cout << "  Degree std-dev:     " << gm.degree_std_dev << "\n";

        std::cout << "\n── Degree distribution ─────────────────────────────\n";
        std::cout << "  Gini coefficient:   " << gm.gini_coefficient << "\n";
        std::cout << "  (0 = perfectly uniform, 1 = one node has all edges)\n";
        std::cout << "  Interpretation:     ";
        if      (gm.gini_coefficient < 0.3) std::cout << "UNIFORM — similar degrees. Good for GPU.\n";
        else if (gm.gini_coefficient < 0.6) std::cout << "MODERATE skew — some hubs exist.\n";
        else                                std::cout << "HIGHLY SKEWED — scale-free/hub structure. Bad for GPU load balance.\n";

        // Log2-bucketed histogram
        std::cout << "\n── Degree histogram (log2 buckets) ─────────────────\n";
        std::vector<int> bucket_counts(32, 0);
        for (int d : degrees) {
            int bucket = (d == 0) ? 0 : std::min(31, (int)std::floor(std::log2(d)) + 1);
            bucket_counts[bucket]++;
        }
        for (int b = 0; b < 32; b++) {
            if (bucket_counts[b] == 0) continue;
            int lo = (b == 0) ? 0 : (1 << (b - 1));
            int hi = (b == 0) ? 0 : (1 << b) - 1;
            double frac    = (double)bucket_counts[b] / n;
            int    bar_len = (int)(frac * 40);
            std::string bar(bar_len, '#');  // ASCII bar — safe on all terminals
            std::cout << "  deg [" << std::setw(6) << lo << " - " << std::setw(6) << hi << "]: "
                      << std::setw(8) << bucket_counts[b] << " nodes  "
                      << std::left << std::setw(42) << bar << std::right
                      << std::fixed << std::setprecision(1) << frac * 100.0 << "%\n";
        }

        std::cout << "\n── Memory & cache behavior ─────────────────────────\n";
        std::cout << "  labels[] array size:     " << gm.label_working_set_mb << " MB\n";
        std::cout << "  Typical CPU L3 cache:    8–32 MB\n";
        std::cout << "  Fits in L3 cache?        "
                  << (fits_in_cache
                      ? "YES  <- CPU cache hides the random access penalty"
                      : "NO   <- cache misses unavoidable; GPU bandwidth may help")
                  << "\n";

        double spread_frac = gm.avg_neighbor_id_spread / n;
        std::cout << "\n  Avg neighbor ID spread:  " << gm.avg_neighbor_id_spread
                  << "  (" << std::fixed << std::setprecision(3) << spread_frac * 100.0 << "% of n)\n";
        std::cout << "  (How scattered a node's neighbors are in the ID space.\n"
                  << "   High -> every labels[neighbor] read hits a different cache line.)\n";
        std::cout << "  Interpretation:          ";
        if      (spread_frac < 0.01) std::cout << "GOOD locality — BFS-like ordering.\n";
        else if (spread_frac < 0.10) std::cout << "MODERATE locality.\n";
        else                         std::cout << "POOR locality — highly scattered IDs. Expect many cache misses.\n";

        std::cout << "\n── GPU-specific ────────────────────────────────────\n";
        std::cout << "  p10 degree:              " << gm.p10_degree << "\n";
        std::cout << "  p90 degree:              " << gm.p90_degree << "\n";
        std::cout << "  Warp efficiency (p10/p90): " << gm.warp_efficiency_estimate << "\n";
        std::cout << "  (How evenly work is split across threads in a GPU warp.\n"
                  << "   Uses p10/p90 instead of min/max so outlier nodes don't distort it.)\n";
        std::cout << "  Interpretation:          ";
        if      (gm.warp_efficiency_estimate > 0.5) std::cout << "GOOD — low warp divergence.\n";
        else if (gm.warp_efficiency_estimate > 0.1) std::cout << "MODERATE — noticeable warp divergence.\n";
        else                                        std::cout << "POOR — hub nodes dominate; most warp threads will idle.\n";

        std::cout << "\n  Arithmetic intensity:    ~" << gm.arithmetic_intensity_estimate << " ops/byte\n";
        std::cout << "  LPA is always memory-bound (GPU tensor core peak needs ~100x this).\n";
        std::cout << "  Winner is determined by memory bandwidth + cache, never by FLOPS.\n";

        std::cout << "\n── Recommendation ──────────────────────────────────\n";
        std::cout << "  -> ";
        switch (gm.recommendation) {
            case GraphMetrics::Recommendation::CPU:     std::cout << "USE CPU\n";                   break;
            case GraphMetrics::Recommendation::CUDA:    std::cout << "USE CUDA\n";                  break;
            case GraphMetrics::Recommendation::UNCLEAR: std::cout << "UNCLEAR — benchmark both\n";  break;
        }
        std::cout << "  " << gm.recommendation_reason << "\n";
        std::cout << "════════════════════════════════════════════════════\n\n";
    }

    return gm;
}