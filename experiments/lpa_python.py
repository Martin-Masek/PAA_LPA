#!/usr/bin/env python3
"""
Label Propagation Algorithm — Python comparison script.
Runs LPA with both NetworkX and igraph and prints timing results
in the same format as the CUDA version:
    RESULT <nodes> <communities> <seconds>

Usage:
    python3 lpa_python.py <edge_list_file> [max_nodes]
"""

import sys
import time

def load_graph(path, max_nodes=-1):
    """Load edge list using the same logic as the C++ loader."""
    remap = {}
    edges = []
    next_id = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u_orig, v_orig = int(parts[0]), int(parts[1])

            new_nodes = (u_orig not in remap) + (v_orig not in remap)
            if max_nodes > 0 and next_id + new_nodes > max_nodes:
                break

            if u_orig not in remap:
                remap[u_orig] = next_id; next_id += 1
            if v_orig not in remap:
                remap[v_orig] = next_id; next_id += 1

            u, v = remap[u_orig], remap[v_orig]
            if u != v:
                edges.append((u, v))

    print(f"Loaded {next_id} nodes, {len(edges)} directed edges")
    return next_id, edges


def run_networkx(n_nodes, edges):
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges)

    t0 = time.perf_counter()
    communities = list(nx.algorithms.community.label_propagation_communities(G))
    elapsed = time.perf_counter() - t0

    n_comm = len(communities)
    print(f"RESULT {n_nodes} {n_comm} {elapsed:.6f}  [networkx]")


def run_igraph(n_nodes, edges):
    import igraph as ig

    G = ig.Graph(n=n_nodes, edges=edges, directed=False)

    t0 = time.perf_counter()
    membership = G.community_label_propagation()
    elapsed = time.perf_counter() - t0

    n_comm = len(set(membership.membership))
    print(f"RESULT {n_nodes} {n_comm} {elapsed:.6f}  [igraph]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lpa_python.py <edge_list_file> [max_nodes]")
        sys.exit(1)

    path = sys.argv[1]
    max_nodes = int(sys.argv[2]) if len(sys.argv) >= 3 else -1

    n_nodes, edges = load_graph(path, max_nodes)

    run_networkx(n_nodes, edges)
    run_igraph(n_nodes, edges)
