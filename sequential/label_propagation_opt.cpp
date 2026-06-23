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

struct CSRGraph {
    int n_nodes;
    std::vector<int> offsets;
    std::vector<int> neighbors;
    std::vector<float> weights;
};

CSRGraph load_edge_list(const std::string& path, bool weighted = false, int max_nodes = -1) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << path << "\n";
        exit(1);
    }

    std::vector<std::tuple<int,int,float>> edges;
    std::map<int,int> node_remap;
    int next_id = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int u, v;
        float w = 1.0f;
        if (!(iss >> u >> v)) continue;
        if (weighted) iss >> w;

        int new_nodes = (node_remap.find(u) == node_remap.end() ? 1 : 0)
                      + (node_remap.find(v) == node_remap.end() ? 1 : 0);
        if (max_nodes > 0 && next_id + new_nodes > max_nodes) break;

        if (node_remap.find(u) == node_remap.end()) node_remap[u] = next_id++;
        if (node_remap.find(v) == node_remap.end()) node_remap[v] = next_id++;

        int ru = node_remap[u];
        int rv = node_remap[v];
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

    for (auto& [u, v, w] : edges)
        g.offsets[u + 1]++;
    for (int i = 1; i <= n; i++)
        g.offsets[i] += g.offsets[i-1];

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

std::vector<int> label_propagation(const CSRGraph& g, int max_iter = 1000,
                                    unsigned seed = 42) {
    std::vector<int> labels(g.n_nodes);
    std::iota(labels.begin(), labels.end(), 0);
    std::vector<int> new_labels(g.n_nodes);
    std::mt19937 rng(seed);

    // FIX 1: single scores map reused across all nodes.
    std::unordered_map<int, float> scores;
    scores.reserve(64);

    for (int iter = 0; iter < max_iter; iter++) {
        int n_changed = 0;

        for (int v = 0; v < g.n_nodes; v++) {
            int nbr_start = g.offsets[v];
            int nbr_end   = g.offsets[v + 1];

            if (nbr_start == nbr_end) {
                new_labels[v] = labels[v];
                continue;
            }

            scores.clear();
            for (int i = nbr_start; i < nbr_end; i++)
                scores[labels[g.neighbors[i]]] += g.weights[i];

            float best_score = -1.0f;
            int   tie_count  = 0;
            for (auto& [lbl, score] : scores) {
                if (score > best_score) { best_score = score; tie_count = 1; }
                else if (score == best_score) tie_count++;
            }
            int   pick       = rng() % tie_count;
            int   best_label = labels[v];
            int   i          = 0;
            for (auto& [lbl, score] : scores) {
                if (score == best_score && i++ == pick) { best_label = lbl; break; }
            }

            new_labels[v] = best_label;
            if (best_label != labels[v]) n_changed++;
        }

        labels.swap(new_labels);

        std::cout << "Iteration " << iter + 1
                  << ": " << n_changed << " nodes changed\n";

        if (n_changed == 0) {
            std::cout << "Converged.\n";
            break;
        }
    }
    return labels;
}

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
        g.n_nodes = 6;
        g.offsets   = {0, 2, 4, 6, 8, 10, 12};
        g.neighbors = {1,2, 0,2, 0,1, 4,5, 3,5, 3,4};
        g.weights   = {1,1, 1,1, 1,1, 1,1, 1,1, 1,1};
    }

    print_graph_stats(g);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto labels = label_propagation(g, 1000, 42);
    auto t1 = std::chrono::high_resolution_clock::now();
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
