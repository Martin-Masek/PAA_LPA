#include <vector>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <chrono>

// ---------------------------------------------------------------------------
// CSR (Compressed Sparse Row) graph representation
//
// Stores the adjacency list of an undirected graph in two flat arrays:
//   offsets[v]..offsets[v+1]  — half-open range of neighbor indices for node v
//   neighbors[i]              — target node of the i-th edge
//   weights[i]                — weight of the i-th edge (1.0 for unweighted)
//
// Memory layout is cache-friendly for sequential traversal of one node's
// neighbors, but accessing neighbors of different nodes in parallel causes
// irregular (non-coalesced) memory accesses — a key motivation for GPU
// optimizations.
// ---------------------------------------------------------------------------
struct CSRGraph {
    int n_nodes;
    std::vector<int>   offsets;
    std::vector<int>   neighbors;
    std::vector<float> weights;
};

// Loads an edge list file into a CSRGraph.
// Format: one edge per line — "u v" or "u v weight", '#' lines are comments.
// Handles non-contiguous / non-zero-based node IDs via remapping.
// Builds an undirected graph (both edge directions stored).
CSRGraph load_edge_list(const std::string& path, bool weighted = false, int max_nodes = -1) {
    std::ifstream file(path);
    if (!file.is_open()) { std::cerr << "Error: cannot open " << path << "\n"; exit(1); }

    std::vector<std::tuple<int,int,float>> edges;
    std::map<int,int> node_remap;
    int next_id = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int u, v; float w = 1.0f;
        if (!(iss >> u >> v)) continue;
        if (weighted) iss >> w;

        int new_nodes = (node_remap.find(u) == node_remap.end() ? 1 : 0)
                      + (node_remap.find(v) == node_remap.end() ? 1 : 0);
        if (max_nodes > 0 && next_id + new_nodes > max_nodes) break;

        if (node_remap.find(u) == node_remap.end()) node_remap[u] = next_id++;
        if (node_remap.find(v) == node_remap.end()) node_remap[v] = next_id++;

        int ru = node_remap[u], rv = node_remap[v];
        if (ru == rv) continue;
        edges.push_back({ru, rv, w});
        edges.push_back({rv, ru, w});
    }

    int n = next_id;
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    CSRGraph g;
    g.n_nodes = n;
    g.offsets.resize(n + 1, 0);
    for (auto& [u, v, w] : edges) g.offsets[u + 1]++;
    for (int i = 1; i <= n; i++)  g.offsets[i] += g.offsets[i-1];
    g.neighbors.resize(edges.size());
    g.weights.resize(edges.size());
    std::vector<int> cursor = g.offsets;
    for (auto& [u, v, w] : edges) {
        g.neighbors[cursor[u]] = v;
        g.weights[cursor[u]++] = w;
    }
    return g;
}

void print_graph_stats(const CSRGraph& g) {
    int   max_deg = 0;
    float avg_deg = 0;
    for (int v = 0; v < g.n_nodes; v++) {
        int d = g.offsets[v+1] - g.offsets[v];
        max_deg = std::max(max_deg, d);
        avg_deg += d;
    }
    std::cout << "Graph: " << g.n_nodes << " nodes, "
              << g.neighbors.size() / 2 << " edges, "
              << "max degree " << max_deg << ", "
              << "avg degree " << avg_deg / g.n_nodes << "\n\n";
}

// ---------------------------------------------------------------------------
// Label Propagation Algorithm — synchronous (double-buffered) variant
//
// Algorithm overview:
//   Initialisation: each node v gets a unique label  labels[v] = v
//
//   Each iteration consists of three sequential phases:
//
//   Phase 1 — Label vote counting  (dominates runtime, O(m) per iteration)
//     For every node v, traverse all neighbours and accumulate the total
//     weight of votes for each distinct label seen among them.
//     This is stored in a per-node hash map: label → accumulated weight.
//     → This phase is EMBARRASSINGLY PARALLEL: each node's computation
//       reads only labels[] (shared, read-only this iteration) and writes
//       only new_labels[v] (private slot). No data races exist.
//       The GPU implementation replaces this loop with parallelFor<Cuda>.
//
//   Phase 2 — Best-label selection  (O(distinct labels per node))
//     Pick the label with the highest accumulated weight. Break ties by
//     random selection (one rng() draw among tied candidates).
//     → Also per-node independent; parallelised on GPU alongside Phase 1.
//
//   Phase 3 — Synchronous buffer swap  (O(n))
//     Swap labels ↔ new_labels so every node sees the same label snapshot
//     for the next iteration (synchronous / Jacobi-style update).
//     → Sequential O(n) memcpy; on GPU done via forAllElements<Cuda>.
//
//   Convergence: stop when no node changed its label, or max_iter reached.
//
// Complexity: O(max_iter × m)  where m = number of edges
// ---------------------------------------------------------------------------
std::vector<int> label_propagation(const CSRGraph& g, int max_iter = 100,
                                    unsigned seed = 42) {
    std::vector<int> labels(g.n_nodes);
    std::iota(labels.begin(), labels.end(), 0);   // labels[v] = v
    std::vector<int> new_labels(g.n_nodes);
    std::mt19937 rng(seed);

    using Clock = std::chrono::high_resolution_clock;
    using Dur   = std::chrono::duration<double>;
    Dur t_phase1{0}, t_phase2{0}, t_phase3{0};

    for (int iter = 0; iter < max_iter; iter++) {
        int n_changed = 0;

        // ── Phase 1 & 2: per-node label vote + selection ─────────────────────
        // These two phases are fused into one node loop for cache efficiency.
        // On GPU both are inside a single parallelFor kernel.
        auto p1_start = Clock::now();

        for (int v = 0; v < g.n_nodes; v++) {
            int nbr_start = g.offsets[v];
            int nbr_end   = g.offsets[v + 1];

            if (nbr_start == nbr_end) {
                new_labels[v] = labels[v];
                continue;
            }

            // Phase 1: accumulate neighbour label votes
            std::unordered_map<int, float> scores;
            for (int i = nbr_start; i < nbr_end; i++)
                scores[labels[g.neighbors[i]]] += g.weights[i];

            auto p2_start = Clock::now();
            t_phase1 += p2_start - p1_start;

            // Phase 2: find best label, random tie-breaking
            float best_score = -1.0f;
            std::vector<int> candidates;
            for (auto& [lbl, score] : scores) {
                if (score > best_score) {
                    best_score = score;
                    candidates.clear();
                    candidates.push_back(lbl);
                } else if (score == best_score) {
                    candidates.push_back(lbl);
                }
            }
            int best_label = candidates[rng() % candidates.size()];
            new_labels[v]  = best_label;
            if (best_label != labels[v]) n_changed++;

            p1_start = Clock::now();
            t_phase2 += p1_start - p2_start;
        }

        // ── Phase 3: synchronous buffer swap ─────────────────────────────────
        auto p3_start = Clock::now();
        labels.swap(new_labels);
        t_phase3 += Clock::now() - p3_start;

        std::cout << "Iteration " << iter + 1
                  << ": " << n_changed << " nodes changed\n";

        if (n_changed == 0) { std::cout << "Converged.\n"; break; }
    }

    std::cout << "\n=== Phase timing (total across all iterations) ===\n";
    std::cout << "  Phase 1 — label vote counting:  " << t_phase1.count() << " s\n";
    std::cout << "  Phase 2 — best-label selection: " << t_phase2.count() << " s\n";
    std::cout << "  Phase 3 — buffer swap:          " << t_phase3.count() << " s\n";
    std::cout << "==================================================\n\n";

    return labels;
}

// Remaps arbitrary label IDs to a clean 0..k-1 range for output.
std::vector<int> normalize_labels(const std::vector<int>& labels) {
    std::unordered_map<int, int> remap;
    int next_id = 0;
    std::vector<int> result(labels.size());
    for (int i = 0; i < (int)labels.size(); i++) {
        if (remap.find(labels[i]) == remap.end())
            remap[labels[i]] = next_id++;
        result[i] = remap[labels[i]];
    }
    return result;
}

int main(int argc, char** argv) {
    CSRGraph g;
    if (argc > 1) {
        int max_nodes = (argc >= 3) ? std::stoi(argv[2]) : -1;
        g = load_edge_list(argv[1], false, max_nodes);
    } else {
        // Minimal test: two triangles, expected 2 communities
        g.n_nodes = 6;
        g.offsets   = {0, 2, 4, 6, 8, 10, 12};
        g.neighbors = {1,2, 0,2, 0,1, 4,5, 3,5, 3,4};
        g.weights   = {1,1, 1,1, 1,1, 1,1, 1,1, 1,1};
    }

    print_graph_stats(g);

    auto t0     = std::chrono::high_resolution_clock::now();
    auto labels = label_propagation(g, 100, 42);
    auto t1     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;

    auto norm  = normalize_labels(labels);
    int  count = *std::max_element(norm.begin(), norm.end()) + 1;

    std::cout << "RESULT "
              << g.n_nodes << " "
              << count << " "
              << elapsed.count()
              << std::endl;
    return 0;
}
