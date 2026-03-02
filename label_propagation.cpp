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
#include <unordered_set>

// CSR (Compressed Sparse Row) graph representation.
// Efficient for graph traversal: neighbors of node v are
// g.neighbors[ g.offsets[v] .. g.offsets[v+1] )
struct CSRGraph {
    int n_nodes;
    std::vector<int> offsets;    // offsets[v]..offsets[v+1] = neighbor range for node v
    std::vector<int> neighbors;  // flat list of neighbor node IDs
    std::vector<float> weights;  // edge weights (1.0 for unweighted graphs)
};

// Loads an edge list file into a CSRGraph.
// Format: one edge per line, "u v" or "u v weight"
// Lines starting with '#' are treated as comments.
// Handles non-contiguous or non-zero-based node IDs via remapping.
// Builds an undirected graph (both directions stored).
CSRGraph load_edge_list(const std::string& path, bool weighted = false) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << path << "\n";
        exit(1);
    }

    std::vector<std::tuple<int,int,float>> edges;
    std::map<int,int> node_remap;  // maps original node IDs -> contiguous 0..n-1 IDs
    int next_id = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int u, v;
        float w = 1.0f;

        if (!(iss >> u >> v)) continue;
        if (weighted) iss >> w;

        // Assign new contiguous IDs to any previously unseen nodes
        if (node_remap.find(u) == node_remap.end()) node_remap[u] = next_id++;
        if (node_remap.find(v) == node_remap.end()) node_remap[v] = next_id++;

        int ru = node_remap[u];
        int rv = node_remap[v];

        if (ru == rv) continue;  // Skip self-loops

        // Add both directions to make the graph undirected
        edges.push_back({ru, rv, w});
        edges.push_back({rv, ru, w});
    }

    int n = next_id;

    // Sort by source node — required to build CSR format correctly
    std::sort(edges.begin(), edges.end());

    // Remove duplicate edges (some datasets list edges twice)
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    // --- Build CSR structure ---
    CSRGraph g;
    g.n_nodes = n;
    g.offsets.resize(n + 1, 0);
    g.neighbors.reserve(edges.size());
    g.weights.reserve(edges.size());

    // Count how many neighbors each node has (stored in offsets[v+1])
    for (auto& [u, v, w] : edges)
        g.offsets[u + 1]++;

    // Convert counts to prefix sums — offsets[v] = start index of v's neighbors
    for (int i = 1; i <= n; i++)
        g.offsets[i] += g.offsets[i-1];

    // Fill neighbor/weight arrays using a cursor per node to track insert position
    g.neighbors.resize(edges.size());
    g.weights.resize(edges.size());
    std::vector<int> cursor = g.offsets;

    for (auto& [u, v, w] : edges) {
        g.neighbors[cursor[u]] = v;
        g.weights[cursor[u]]   = w;
        cursor[u]++;
    }

    return g;
}

void print_graph_stats(const CSRGraph& g) {
    int max_degree = 0;
    float avg_degree = 0;
    for (int v = 0; v < g.n_nodes; v++) {
        int deg = g.offsets[v+1] - g.offsets[v];
        max_degree = std::max(max_degree, deg);
        avg_degree += deg;
    }
    avg_degree /= g.n_nodes;

    std::cout << "Graph loaded:\n";
    std::cout << "  Nodes:      " << g.n_nodes << "\n";
    std::cout << "  Edges:      " << g.neighbors.size() / 2 << "\n";
    std::cout << "  Max degree: " << max_degree << "\n";
    std::cout << "  Avg degree: " << avg_degree << "\n\n";
}

// Label Propagation Algorithm (LPA) for community detection.
//
// Each node starts with a unique label (its own ID).
// In each iteration, every node adopts the label most common among its
// neighbors (weighted by edge weight). Ties are broken randomly.
// Updates are synchronous: all nodes read old labels, write to new_labels,
// then swap — this is the synchronous variant of LPA.
// Stops early if no label changes in a full iteration.
//
// Returns a vector where labels[v] = community ID for node v.
std::vector<int> label_propagation(const CSRGraph& g, int max_iter = 100,
                                    unsigned seed = 42, int debug_node_count = 3) {
    std::vector<int> labels(g.n_nodes);
    std::iota(labels.begin(), labels.end(), 0);
    std::vector<int> new_labels(g.n_nodes);
    std::mt19937 rng(seed);

    for (int iter = 0; iter < max_iter; iter++) {
        bool changed    = false;
        int  n_changed  = 0;

        for (int v = 0; v < g.n_nodes; v++) {
            int nbr_start = g.offsets[v];
            int nbr_end   = g.offsets[v + 1];

            if (nbr_start == nbr_end) {
                new_labels[v] = labels[v];
                continue;
            }

            std::unordered_map<int, float> scores;
            for (int i = nbr_start; i < nbr_end; i++)
                scores[labels[g.neighbors[i]]] += g.weights[i];

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

            int best_label    = candidates[rng() % candidates.size()];
            new_labels[v]     = best_label;
            if (best_label != labels[v]) { changed = true; n_changed++; }
        }

        labels.swap(new_labels);

        // Count communities
        std::unordered_set<int> unique(labels.begin(), labels.end());

        std::cout << "\n=== Iteration " << iter + 1 << " ===\n";
        std::cout << "  Communities: " << unique.size() << "\n";
        std::cout << "  Nodes changed: " << n_changed << " / " << g.n_nodes << "\n";

        if (!changed) {
            std::cout << "  Converged — no nodes changed.\n";
            break;
        }
    }
    return labels;
}

// Remaps community labels to a clean 0, 1, 2, ... range.
// LPA labels are arbitrary node IDs; this makes them contiguous for output.
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
        g = load_edge_list(argv[1]);  // Load graph from edge list file
    } else {
        // Hardcoded fallback: two triangles (0-1-2) and (3-4-5), no cross edges
        // Expected result: 2 communities
        g.n_nodes = 6;
        g.offsets   = {0, 2, 4, 6, 8, 10, 12};
        g.neighbors = {1,2, 0,2, 0,1, 4,5, 3,5, 3,4};
        g.weights   = {1,1, 1,1, 1,1, 1,1, 1,1, 1,1};
    }

    print_graph_stats(g);

    // Run LPA with multiple random seeds and keep the result with fewest communities
    // (fewer communities = more consolidated, often better quality)
    int best_count = std::numeric_limits<int>::max();
    std::vector<int> best_labels;

    for (unsigned seed : {42u}) {
        auto labels = label_propagation(g, 100, seed);
        auto norm   = normalize_labels(labels);
        int count   = *std::max_element(norm.begin(), norm.end()) + 1;
        std::cout << "Seed " << seed << " -> " << count << " communities\n";
        if (count < best_count) {
            best_count  = count;
            best_labels = norm;
        }
    }

    std::cout << "Best: " << best_count << " communities\n";
    return 0;
}