#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <chrono>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>

using RealType  = float;
using IndexType = int;

// Host-side loading always uses CSR — SlicedEllpack::setRowCapacities calls
// TNL::sum internally which is not defined for Host arrays in this TNL build.
using HostMatrix = TNL::Matrices::SparseMatrix<RealType, TNL::Devices::Host, IndexType>;

// Device matrix uses BiEllpack: sorts rows within each warp-sized slice by
// degree before padding, so high- and low-degree nodes are grouped together.
// This minimises padding waste on power-law graphs compared to SlicedEllpack.
using DeviceMatrix = TNL::Matrices::SparseMatrix<
    RealType, TNL::Devices::Cuda, IndexType,
    TNL::Matrices::GeneralMatrix,
    TNL::Algorithms::Segments::BiEllpack>;

DeviceMatrix load_to_tnl_matrix(const std::string& path, bool weighted = false, int max_nodes = -1) {
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

    HostMatrix host_matrix;
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
    DeviceMatrix device_matrix;
    device_matrix = host_matrix;
    return device_matrix;
}

void checksum_matrix(const HostMatrix& m) {
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

// ─── LPA ─────────────────────────────────────────────────────────────────────

TNL::Containers::Array<int, TNL::Devices::Host>
run_lpa(DeviceMatrix& matrix, int max_iter = 1000) {
    int n = matrix.getRows();

    TNL::Containers::Array<int, TNL::Devices::Cuda> labels(n);
    TNL::Containers::Array<int, TNL::Devices::Cuda> new_labels(n);
    TNL::Containers::Array<int, TNL::Devices::Cuda> changed_flags(n);

    labels.forAllElements(
        [] __cuda_callable__ (int i, int& value) { value = i; });
    new_labels.forAllElements(
        [] __cuda_callable__ (int i, int& value) { value = i; });
    changed_flags.forAllElements(
        [] __cuda_callable__ (int i, int& value) { value = 0; });

    auto labels_view     = labels.getView();
    auto new_labels_view = new_labels.getView();
    auto changed_view    = changed_flags.getView();
    auto matrix_view     = matrix.getView();

    for (int iter = 0; iter < max_iter; iter++) {

        TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, n,
            [=] __cuda_callable__ (int v) mutable {
                auto row = matrix_view.getRow(v);

                const int MAX_LOCAL_LABELS = 32;
                int   label_ids   [MAX_LOCAL_LABELS];
                float label_scores[MAX_LOCAL_LABELS];
                int   num_labels = 0;

                // SlicedEllpack pads rows to the slice max length; padding
                // elements have column index -1 and must be skipped.
                for (int i = 0; i < row.getSize(); i++) {
                    int neighbor = row.getColumnIndex(i);
                    if (neighbor < 0) continue;
                    int   lbl    = labels_view[neighbor];
                    float weight = row.getValue(i);

                    int idx = -1;
                    for (int j = 0; j < num_labels; j++) {
                        if (label_ids[j] == lbl) { idx = j; break; }
                    }
                    if (idx >= 0) {
                        label_scores[idx] += weight;
                    } else if (num_labels < MAX_LOCAL_LABELS) {
                        label_ids   [num_labels] = lbl;
                        label_scores[num_labels] = weight;
                        num_labels++;
                    }
                }

                if (num_labels == 0) {
                    new_labels_view[v] = labels_view[v];
                    changed_view[v]    = 0;
                    return;
                }

                int   best_label = labels_view[v];
                float best_score = -1.0f;

                for (int i = 0; i < num_labels; i++) {
                    unsigned h = (unsigned(v)            * 2654435761u)
                               ^ (unsigned(label_ids[i]) * 2246822519u)
                               ^ (unsigned(iter)         * 1234567891u);
                    float final_score = label_scores[i] + (float)(h & 0xFFFF) / 1e7f;

                    if (final_score > best_score) {
                        best_score = final_score;
                        best_label = label_ids[i];
                    }
                }

                new_labels_view[v] = best_label;
                changed_view[v]    = (best_label != labels_view[v]) ? 1 : 0;
            });

        int total_changed = TNL::Algorithms::reduce(changed_flags, TNL::Plus{});

        labels.forAllElements(
            [=] __cuda_callable__ (int i, int& value) {
                value = new_labels_view[i];
            });

        if (iter % 10 == 0 || total_changed == 0) {
            std::cout << "Iteration " << iter << ": " << total_changed << " nodes changed.\n";
        }

        if (total_changed == 0) break;
    }

    TNL::Containers::Array<int, TNL::Devices::Host> host_labels;
    host_labels = labels;
    return host_labels;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./tlpa_ellpack <edge_list_file> [max_rows]\n";
        return 1;
    }

    int max_rows = -1;
    if (argc >= 3)
        max_rows = std::stoi(argv[2]);
    try {
        auto matrix = load_to_tnl_matrix(argv[1], false, max_rows);

        HostMatrix host_matrix;
        host_matrix = matrix;
        checksum_matrix(host_matrix);

        std::cout << "Nodes: " << matrix.getRows() << "\n";
        std::cout << "Cols:  " << matrix.getColumns() << "\n";

        auto start = std::chrono::high_resolution_clock::now();

        auto labels = run_lpa(matrix);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::unordered_map<int,int> comm_map;
        int n_comm = 0;
        for (int i = 0; i < labels.getSize(); i++)
            if (comm_map.find(labels[i]) == comm_map.end())
                comm_map[labels[i]] = n_comm++;

        std::cout << "RESULT "
                << matrix.getRows() << " "
                << n_comm << " "
                << elapsed.count()
                << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
