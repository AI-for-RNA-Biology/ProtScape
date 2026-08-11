"""Shared network-layout helpers used across section 8/9 plots."""
import networkx as nx
import numpy as np


def _declump_circles(centers: dict, radii: dict, iterations: int = 200, margin: float = 0.5):
    """Iteratively pushes apart circles that overlap (treating each module as a
    circle of the given radius), so inter-module spacing scales with module
    *size* - not just module *count*. Without this, a spring layout on the
    module-level graph alone can place two large modules' point-clouds close
    enough to visually overlap, which is what made low-resolution partitions
    (few, large modules) look like one big blob regardless of node coloring."""
    ids = list(centers.keys())
    pos = {k: np.array(v, dtype=float) for k, v in centers.items()}
    for _ in range(iterations):
        moved = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                delta = pos[b] - pos[a]
                dist = np.linalg.norm(delta)
                min_dist = radii[a] + radii[b] + margin
                if dist < min_dist:
                    moved = True
                    direction = delta / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    overlap = (min_dist - dist) / 2
                    pos[a] -= direction * overlap
                    pos[b] += direction * overlap
        if not moved:
            break
    return pos


def community_layout(g: nx.Graph, module_of: dict, seed: int = 55) -> dict:
    """Mirrors the R script's two-stage layout: lay out one supernode per
    module (edge-weighted by inter-module edge count, then declumped so
    spacing scales with module size), then lay out each module's internal
    nodes locally and translate them to their supernode's position."""
    mod_ids = sorted(set(module_of.values()))
    module_nodes = {m: [n for n, mm in module_of.items() if mm == m] for m in mod_ids}
    # Same "radius" a module's node-cloud ends up filling below (2 + sqrt(n)),
    # reused here so supernodes are spaced far enough apart not to overlap it.
    module_radius = {m: 2 + np.sqrt(len(nodes)) for m, nodes in module_nodes.items()}

    comm_g = nx.Graph()
    comm_g.add_nodes_from(mod_ids)
    for u, v in g.edges():
        mu, mv = module_of[u], module_of[v]
        if mu != mv:
            a, b = min(mu, mv), max(mu, mv)
            if comm_g.has_edge(a, b):
                comm_g[a][b]["weight"] += 1
            else:
                comm_g.add_edge(a, b, weight=1)

    comm_layout = nx.spring_layout(comm_g, weight="weight", seed=seed)
    scale = max(module_radius.values()) * len(mod_ids) ** 0.5
    comm_layout = {k: v * scale for k, v in comm_layout.items()}
    comm_layout = _declump_circles(comm_layout, module_radius)

    final_layout = {}
    for m in mod_ids:
        nodes = module_nodes[m]
        if len(nodes) == 1:
            local_lay = {nodes[0]: np.zeros(2)}
        else:
            sub_g = g.subgraph(nodes)
            local_lay = nx.spring_layout(sub_g, iterations=500, seed=seed)
            max_extent = max(1e-6, max(np.abs(v).max() for v in local_lay.values()))
            scale_factor = module_radius[m] / max_extent
            local_lay = {k: v * scale_factor for k, v in local_lay.items()}
        center = comm_layout[m]
        for n, pos in local_lay.items():
            final_layout[n] = pos + center
    return final_layout


def module_centroid_positions(module_of: dict, layout: dict) -> dict:
    """Module positions for `plot_module_density_graph`, derived from the
    node-level giant-graph layout (`community_layout()`'s output) rather than
    an independent layout of the module graph - each module is placed at the
    centroid of its own nodes' positions. This keeps the module-density plot
    spatially consistent with where each cluster actually sits on the
    node-level giant plots, and - since it only depends on `layout`, not on
    the module graph's own (known-only vs. known+new) edges - calling this
    once and reusing the result for both variants keeps every module in the
    exact same spot across both."""
    positions = {}
    for n, m in module_of.items():
        positions.setdefault(m, []).append(layout[n])
    return {m: np.mean(pts, axis=0) for m, pts in positions.items()}


def points_per_data_unit(ax) -> float:
    """Approximate points-per-data-unit along x, using the axes' *current* xlim -
    lets a marker radius given in points (as matplotlib's scatter `s` is always in
    points^2) be converted into an equivalent radius in data coordinates."""
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 0))
    dist_pixels = np.hypot(*(p1 - p0))
    return dist_pixels * 72.0 / ax.figure.dpi


def module_layout(giant: nx.Graph, module_of: dict, module_id: int, seed: int = 12) -> dict:
    """Full spring layout of every gene in `module_id` - the same call
    `plot_focus_module()` makes (same default `seed`), so positions here match
    the module views in `modules_oi_*.pdf`. Component plots in
    `edge_components.py` take their genes' positions FROM this instead of
    laying each component out on its own, so a component reads as a zoomed-in
    region of that module's full network rather than an unrelated
    re-arrangement."""
    nodes = [n for n, m in module_of.items() if m == module_id]
    sub_g = giant.subgraph(nodes)
    return nx.spring_layout(sub_g, seed=seed)
