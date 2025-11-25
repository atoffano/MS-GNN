import argparse
import go3
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import cm


def parse_predictions(pred_file, target_id, threshold):
    """Parse the prediction TSV and return a dict of GO terms and their scores for the given target IDs above threshold."""
    go_terms = {}
    with open(pred_file) as f:
        header = next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            target, go_term, score = parts[0], parts[1], parts[2]
            if target == target_id:
                try:
                    score = float(score)
                    if score >= threshold:
                        go_terms[go_term] = score
                except ValueError:
                    continue
    return go_terms


def parse_ground_truth(gt_file, target_id):
    """Parse a ground truth file and return a set of GO terms for the given target ID."""
    go_terms = set()
    with open(gt_file) as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            target, go_term = parts[0], parts[1]
            if target == target_id:
                go_terms = set(go_term.split("; "))
                go_terms.add(go_term)
    return go_terms


def get_ancestors(go_terms):
    """Return all ancestors (including self) for a set of GO terms using go3.

    Uses the go3 package for fast ancestor lookups instead of networkx.
    """
    all_terms = set()
    for go_term in go_terms:
        try:
            ancestors = go3.ancestors(go_term)
            all_terms.update(ancestors)
        except (ValueError, KeyError):
            # Term not found in ontology, skip it
            continue
    return all_terms


def build_subgraph(go_terms):
    """Build a subgraph containing only the GO terms and their ancestors using go3.

    Uses the go3 package for fast ontology traversal and builds a networkx
    DiGraph for visualization purposes.
    """
    nodes = get_ancestors(go_terms)
    subgraph = nx.DiGraph()

    # Add nodes with their attributes from go3
    for node in nodes:
        try:
            term = go3.get_term_by_id(node)
            subgraph.add_node(
                node,
                name=term.name,
                namespace=term.namespace,
            )
        except (ValueError, KeyError):
            # Term not found, add with minimal attributes
            subgraph.add_node(node, name="", namespace="")

    # Add edges (from parent to child for visualization)
    for node in nodes:
        try:
            term = go3.get_term_by_id(node)
            # Add edges from this node to its parents (in go3, parents are the ancestors direction)
            for parent in term.parents:
                if parent in nodes:
                    # Edge from parent to child for hierarchy visualization
                    subgraph.add_edge(parent, node)
        except (ValueError, KeyError):
            continue

    print(
        f"Subgraph contains {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges."
    )
    return subgraph


def get_hierarchy_pos(
    G,
    root=None,
    width=1.0,
    vert_gap=0.2,
    vert_loc=0,
    xcenter=0.5,
):
    if root is None:
        roots = [n for n, d in G.in_degree() if d == 0]
        root = roots[0] if roots else list(G.nodes())[0]
    levels = dict(nx.single_source_shortest_path_length(G, root))
    max_level = max(levels.values())
    level_nodes = {lvl: [] for lvl in range(max_level + 1)}
    for node, lvl in levels.items():
        level_nodes[lvl].append(node)
    y_by_level = {lvl: vert_loc - lvl * vert_gap for lvl in level_nodes}
    order = {0: [root]}
    for lvl in range(1, max_level + 1):
        prev = order[lvl - 1]
        children = []
        used = set()
        for parent in prev:
            for child in sorted(G.successors(parent)):
                if child in level_nodes[lvl] and child not in used:
                    children.append(child)
                    used.add(child)
        for node in sorted(level_nodes[lvl]):
            if node not in used:
                children.append(node)
        order[lvl] = children
    pos = {}
    for lvl, nodes_at_level in order.items():
        n = len(nodes_at_level)
        if n == 1:
            pos[nodes_at_level[0]] = (xcenter, y_by_level[lvl])
        else:
            for i, node in enumerate(nodes_at_level):
                x = xcenter - width / 2 + width * (i + 0.5) / n
                pos[node] = (x, y_by_level[lvl])
    return pos


def plot_multi_pred_gt(
    pred_dicts,
    pred_names,
    pred_cmaps,
    gt_terms,
    subgraph,
    protein_name,
    truth_only=False,
):
    """
    Plot the GO subgraph using a single GT and multiple prediction files.
    Each node is a large circle (gray if in GT, white if not), and for each prediction file,
    a smaller colored subnode is plotted below the main node, colored by method and score.
    """
    # Node shape and size
    node_shape = "o"
    node_size = 1000
    subnode_size = node_size // 4  # half of previous subnode size
    nodes_to_plot = list(subgraph.nodes())

    # Format labels: word-aware wrapping at 20 chars per line
    labels = {}
    for n in nodes_to_plot:
        name = subgraph.nodes[n].get("name", "")
        words = name.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + (1 if current_line else 0) <= 20:
                if current_line:
                    current_line += " "
                current_line += word
            else:
                if current_line:
                    lines.append(current_line)
                while len(word) > 20:
                    lines.append(word[:20])
                    word = word[20:]
                current_line = word
        if current_line:
            lines.append(current_line)
        name_formatted = "\n".join(lines)
        labels[n] = f"{n}\n{name_formatted}"

    # Layout
    roots = ["GO:0008150", "GO:0005575", "GO:0003674"]
    root = None
    for r in roots:
        if r in subgraph:
            root = r
            break
    hier_pos = get_hierarchy_pos(subgraph, root=root)
    pos = {n: (hier_pos[n][0], hier_pos[n][1]) for n in nodes_to_plot}

    plt.figure(figsize=(24, 15))
    ax = plt.gca()
    # Draw edges, curve if nodes are on the same y level
    for u, v in subgraph.edges():
        if u not in nodes_to_plot or v not in nodes_to_plot:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if abs(y1 - y2) < 1e-8:  # same y level
            rad = 0.2 if x2 > x1 else -0.2
            con = mpatches.FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                color="gray",
                linewidth=1.5,
                mutation_scale=10,
                zorder=1,
            )
            ax.add_patch(con)
        elif abs(y1 - y2) >= 1e-8:
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="gray",
                    lw=1.5,
                    shrinkA=10,
                    shrinkB=10,
                ),
                zorder=1,
            )

    # Draw main nodes (GT: gray, not GT: white)
    nodelist_gt = [n for n in nodes_to_plot if n in gt_terms]
    nodelist_notgt = [n for n in nodes_to_plot if n not in gt_terms]
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=nodelist_gt,
        node_color="#cccccc",
        node_size=node_size,
        edgecolors="black",
        linewidths=1,
        node_shape=node_shape,
    )
    if not truth_only:
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=nodelist_notgt,
            node_color="#ffffff",
            node_size=node_size,
            edgecolors="black",
            linewidths=1,
            node_shape=node_shape,
        )

    # Draw subnodes for each prediction file
    y_offset = 0.03

    for n in nodes_to_plot:
        x, y = pos[n]
        # Center subnodes horizontally under the node without gap between them
        start_x = x
        for i, (pred, cmap, name) in enumerate(zip(pred_dicts, pred_cmaps, pred_names)):
            if n in pred:
                score = pred[n]
                color = cmap(mpl.colors.Normalize(vmin=0, vmax=1)(score))

                if i == 0:
                    x_shift = 0.0
                else:
                    x_shift = i * 0.01

                x_off = start_x + x_shift
                y_off = y - y_offset
                ax.scatter(
                    [x_off],
                    [y_off],
                    s=subnode_size,
                    c=[color],
                    edgecolors="black",
                    linewidths=1,
                    marker="o",
                )

    # Draw labels
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels=labels,
        font_size=8,
    )

    plt.title(f"GO graph: predictions (subnodes) & GT (main node) for {protein_name}")
    plt.axis("off")
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave space for colorbar

    # Colorbars for each method
    for i, (cmap, name) in enumerate(zip(pred_cmaps, pred_names)):
        cbar_ax = plt.gcf().add_axes([0.93, 0.2 + 0.08 * i, 0.02, 0.06])
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f"{name} score", fontsize=10)

    # Custom legend for shapes and subnodes
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="In GT",
            markerfacecolor="#cccccc",
            markeredgecolor="black",
            markersize=15,
        ),
    ]
    if not truth_only:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Not in GT",
                markerfacecolor="#ffffff",
                markeredgecolor="black",
                markersize=15,
            )
        )
    for i, name in enumerate(pred_names):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{name} prediction",
                markerfacecolor=cm.get_cmap(pred_cmaps[i])(0.8),
                markeredgecolor="black",
                markersize=6,
            )
        )
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1.08))

    # Save with protein name
    safe_name = "".join([c if c.isalnum() or c in "-_." else "_" for c in protein_name])
    outname = f"go_multi_pred_gt_{safe_name}.png"
    plt.savefig(outname, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved GO multi-prediction+GT graph to {outname}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot GO graph for selected protein: multiple predictions (subnodes) & GT (main node)."
    )
    parser.add_argument(
        "--pred",
        nargs=3,
        action="append",
        metavar=("PRED_TSV", "THRESH", "NAME"),
        help="Prediction TSV file and threshold (can be used multiple times)",
        required=True,
    )
    parser.add_argument(
        "--gt", required=True, help="Ground truth file (target_ID term_ID)"
    )
    parser.add_argument("--obo", required=True, help="GO OBO file")
    parser.add_argument("--protein", required=True, help="Protein target ID to plot")
    parser.add_argument(
        "--truth_only",
        action="store_true",
        help="Only display nodes present in ground truth",
    )
    args = parser.parse_args()

    print("Loading ontology with go3...")
    try:
        go3.load_go_terms(args.obo)
    except Exception as e:
        print(f"Error loading ontology from {args.obo}: {e}")
        return

    print("Parsing ground truth...")
    gt_terms = parse_ground_truth(args.gt, args.protein)
    print(f"Found {len(gt_terms)} terms in GT.")

    pred_dicts = []
    pred_names = []
    pred_cmaps = []
    color_maps = [
        cm.Reds,
        cm.Blues,
        cm.Greens,
        cm.Purples,
        cm.Oranges,
        cm.Greys,
        cm.PiYG,
        cm.cividis,
    ]
    for i, (pred_file, thresh, name) in enumerate(args.pred):
        print(f"Parsing predictions from {pred_file} with threshold {thresh}...")
        pred_dicts.append(parse_predictions(pred_file, args.protein, float(thresh)))
        pred_cmaps.append(color_maps[i % len(color_maps)])
        pred_names.append(name)

    print("Building subgraph...")
    all_terms = set(gt_terms)
    if not args.truth_only:
        for pred in pred_dicts:
            all_terms |= set(pred.keys())
    # If --truth_only, do NOT add predicted terms not in GT
    subgraph = build_subgraph(all_terms)
    print("Plotting...")
    plot_multi_pred_gt(
        pred_dicts,
        pred_names,
        pred_cmaps,
        gt_terms,
        subgraph,
        args.protein,
        truth_only=args.truth_only,
    )


if __name__ == "__main__":
    main()
