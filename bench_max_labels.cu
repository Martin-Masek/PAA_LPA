#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <chrono>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>

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
    std::cerr << "Loaded " << n_nodes << " nodes\n";

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

    std::cerr << "Transferring graph to device...\n";
    GraphMatrix<Device> device_matrix;
    device_matrix = host_matrix;
    return device_matrix;
}

struct BenchResult {
    int    max_local_labels;
    double time_seconds;
    int    num_communities;
    int    iterations_done;
    int    final_changed;
};

template<typename Device, int MAX_LABELS>
BenchResult run_lpa_bench(GraphMatrix<Device>& matrix, int max_iter = 1000) {
    int n = matrix.getRows();

    TNL::Containers::Array<int, Device> labels(n);
    TNL::Containers::Array<int, Device> new_labels(n);
    TNL::Containers::Array<int, Device> changed_flags(n);

    labels.forAllElements(
        [] __cuda_callable__ (int i, int& v) { v = i; });
    new_labels.forAllElements(
        [] __cuda_callable__ (int i, int& v) { v = i; });
    changed_flags.forAllElements(
        [] __cuda_callable__ (int i, int& v) { v = 0; });

    auto labels_view     = labels.getView();
    auto new_labels_view = new_labels.getView();
    auto changed_view    = changed_flags.getView();
    auto matrix_view     = matrix.getView();

    cudaDeviceSynchronize();
    auto t_start = std::chrono::high_resolution_clock::now();

    int iterations_done = 0;
    int final_changed   = n;

    for (int iter = 0; iter < max_iter; iter++) {
        TNL::Algorithms::parallelFor<Device>(0, n,
            [=] __cuda_callable__ (int v) mutable {
                auto row   = matrix_view.getRow(v);
                int degree = row.getSize();

                if (degree == 0) {
                    new_labels_view[v] = labels_view[v];
                    changed_view[v]    = 0;
                    return;
                }

                int   label_ids   [MAX_LABELS];
                float label_scores[MAX_LABELS];
                int   num_labels = 0;

                for (int i = 0; i < degree; i++) {
                    int   neighbor = row.getColumnIndex(i);
                    int   lbl      = labels_view[neighbor];
                    float weight   = row.getValue(i);

                    int idx = -1;
                    for (int j = 0; j < num_labels; j++) {
                        if (label_ids[j] == lbl) { idx = j; break; }
                    }
                    if (idx >= 0) {
                        label_scores[idx] += weight;
                    } else if (num_labels < MAX_LABELS) {
                        label_ids   [num_labels] = lbl;
                        label_scores[num_labels] = weight;
                        num_labels++;
                    }
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

        final_changed = TNL::Algorithms::reduce(changed_flags, TNL::Plus{});

        labels.forAllElements(
            [=] __cuda_callable__ (int i, int& value) {
                value = new_labels_view[i];
            });

        iterations_done = iter + 1;
        if (final_changed == 0) break;
    }

    cudaDeviceSynchronize();
    auto t_end = std::chrono::high_resolution_clock::now();

    TNL::Containers::Array<int, TNL::Devices::Host> host_labels;
    host_labels = labels;

    std::unordered_map<int,int> comm_map;
    int n_comm = 0;
    for (int i = 0; i < host_labels.getSize(); i++)
        if (!comm_map.count(host_labels[i]))
            comm_map[host_labels[i]] = n_comm++;

    return BenchResult{
        MAX_LABELS,
        std::chrono::duration<double>(t_end - t_start).count(),
        n_comm,
        iterations_done,
        final_changed
    };
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./bench_max_labels <edge_list_file> [max_nodes] [max_iter]\n";
        return 1;
    }

    int max_nodes = (argc >= 3) ? std::stoi(argv[2]) : -1;
    int max_iter  = (argc >= 4) ? std::stoi(argv[3]) : 1000;

    try {
        using Device = TNL::Devices::Cuda;
        auto matrix = load_to_tnl_matrix<Device>(argv[1], false, max_nodes);

        std::cout << "max_labels,time_s,communities,iterations,final_changed\n";
        std::cout.flush();

        auto print = [](const BenchResult& r) {
            std::cerr << "  MAX_LABELS=" << r.max_local_labels
                      << "  time=" << r.time_seconds << "s"
                      << "  comms=" << r.num_communities
                      << "  iters=" << r.iterations_done
                      << "  remaining=" << r.final_changed << "\n";
            std::cout << r.max_local_labels << ","
                      << r.time_seconds    << ","
                      << r.num_communities << ","
                      << r.iterations_done << ","
                      << r.final_changed   << "\n";
            std::cout.flush();
        };

#define RUN(N) { std::cerr << "Running MAX_LABELS=" #N "...\n"; print(run_lpa_bench<Device, N>(matrix, max_iter)); }
        RUN(4)
        RUN(8)
        RUN(16)
        RUN(32)
        RUN(64)
        RUN(128)
        RUN(256)
        RUN(512)
        RUN(1024)
#undef RUN

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
