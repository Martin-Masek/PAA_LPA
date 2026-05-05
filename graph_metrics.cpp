#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include "graph_metrics.h"
using RealType  = float;
using IndexType = int;

template<typename Device>
using GraphMatrix = TNL::Matrices::SparseMatrix<RealType, Device, IndexType>;

template<typename Device>
GraphMatrix<Device> load_to_tnl_matrix(const std::string& path, bool weighted = false, int max_nodes = -1) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + path);

    using Edge = std::tuple<IndexType, IndexType, float>;
    std::vector<Edge> edges;
    std::map<IndexType, IndexType> remap;
    IndexType next_id = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        IndexType u_orig, v_orig;
        float w = 1.0f;
        if (!(iss >> u_orig >> v_orig)) continue;
        if (weighted) iss >> w;

        int new_nodes = (!remap.count(u_orig) ? 1 : 0)
                      + (!remap.count(v_orig) ? 1 : 0);
        if (max_nodes > 0 && (int)(next_id + new_nodes) > max_nodes)
            break;

        if (!remap.count(u_orig)) remap[u_orig] = next_id++;
        if (!remap.count(v_orig)) remap[v_orig] = next_id++;
        IndexType u = remap[u_orig];
        IndexType v = remap[v_orig];
        if (u == v) continue;

        edges.push_back({u, v, w});
        edges.push_back({v, u, w});
    }

    IndexType n_nodes = next_id;
    std::cout << "Loaded " << n_nodes << " nodes\n";

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    std::vector<IndexType> row_capacity(n_nodes, 0);
    for (auto& [u, v, w] : edges)
        row_capacity[u]++;

    GraphMatrix<TNL::Devices::Host> host_matrix;
    host_matrix.setDimensions(n_nodes, n_nodes);

    TNL::Containers::Array<IndexType, TNL::Devices::Host> capacities(n_nodes);
    for (IndexType i = 0; i < n_nodes; i++) capacities[i] = row_capacity[i];
    host_matrix.setRowCapacities(capacities);

    std::vector<IndexType> cursor(n_nodes, 0);
    for (auto& [u, v, w] : edges) {
        auto row = host_matrix.getRow(u);
        row.setElement(cursor[u]++, v, w);
    }

    std::cout << "Transferring graph to device...\n";
    GraphMatrix<Device> device_matrix;
    device_matrix = host_matrix;
    return device_matrix;
}

void checksum_matrix(const GraphMatrix<TNL::Devices::Host>& m) {
    long long neighbor_sum = 0, edge_count = 0;
    for (int v = 0; v < m.getRows(); v++) {
        auto row = m.getRow(v);
        for (int i = 0; i < row.getSize(); i++) {
            if (row.getValue(i) == 0.0f) continue;
            neighbor_sum += row.getColumnIndex(i);
            edge_count++;
        }
    }
    std::cout << "=== GRAPH CHECKSUM ===\n";
    std::cout << "  Nodes:        " << m.getRows() << "\n";
    std::cout << "  Edges:        " << edge_count / 2 << "\n";
    std::cout << "  Neighbor sum: " << neighbor_sum << "\n";
    std::cout << "======================\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./tlpa <edge_list_file> [max_rows]\n";
        return 1;
    }

    std::string input_file = argv[1];

    int max_rows = -1;
    if (argc >= 3)
        max_rows = std::stoi(argv[2]);
    try {
        using TargetDevice = TNL::Devices::Host;

        auto matrix = load_to_tnl_matrix<TargetDevice>(argv[1], false, max_rows);

        GraphMatrix<TNL::Devices::Host> host_matrix;
        host_matrix = matrix;

        checksum_matrix(host_matrix);
        auto gm = compute_graph_metrics(host_matrix);  // <-- add this line

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}