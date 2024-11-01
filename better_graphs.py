import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict


def print_graph(graph):
    # Compute the tree-like layout using Graphviz
    pos = nx.drawing.nx_pydot.graphviz_layout(graph, prog='dot', root=list(node_dict.keys())[0])

    # Draw and display the graph
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, arrows=True)
    plt.axis('off')
    plt.show()


def de_alias(node_dict):
    for n in node_dict['names'].keys():
        for _n in node_dict['names'].keys():
            if n == _n: continue
            assert len(set(node_dict['names'][n]).intersection(set(node_dict['names'][_n]))) == 0,\
                f"Aliased nodes must not have overlapping components."

    new_node_dict = dict()
    for k in node_dict.keys():
        if k == 'names':
            continue
        if isinstance(k, int):
            new_node_dict[k] = node_dict[k]
            continue

        parents = set()
        independent = False
        for p_alias in node_dict[k]:
            if p_alias is None:
                independent = True
                continue
            parents = parents.union(set(node_dict['names'][p_alias]))
        parents = sorted(list(parents))

        components = sorted(node_dict['names'][k])
        for i, c in enumerate(components):
            new_node_dict[c] = parents if independent else sorted(parents + components[i+1:])

    return new_node_dict


def process_graph(node_dict, show_graph: bool = False):
    if not isinstance(node_dict, OrderedDict):
        print("[WARN] Processing unordered dict. Nondeterministic parent conflict resolution.")

    if 'names' in node_dict.keys():
        node_dict = de_alias(node_dict)


    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes and edges to the graph based on the dictionary
    for node, parents in node_dict.items():
        graph.add_node(node)
        for parent in parents:
            graph.add_edge(parent, node)

    # Check for cycles in the graph
    if not nx.is_directed_acyclic_graph(graph):
        print_graph(graph)
        raise Exception(f"The graph contains cycles.")

    if show_graph:
        print_graph(graph)

    # Convert the dictionary to a list ordered by keys
    ordered_list = [node_dict[key] for key in sorted(node_dict.keys())]

    return ordered_list


if __name__ == "__main__":
    node_dict = {
        0: [],
        1: [0, 6, 7, 8, 9, 10],
        2: [0, 6, 7, 8, 9, 10],
        3: [0, 6, 7, 8, 9, 10],
        4: [0, 6, 7, 8, 9, 10],
        5: [0, 6, 7, 8, 9, 10],
        6: [],
        7: [6],
        8: [6, 7],
        9: [6, 7, 8],
        10: [6, 7, 8, 9],
    }

    aliased = {
        'names': {
            'theta': [0],
            'X': [1,2,3,4,5],
            'Xp': [6,7,8,9,10]
        },

        'theta': [],
        'X': ['theta', None],  # The None indicates independence within 'X' conditional on the specified parents
        'Xp': ['X'],
        11: [8],
        12: [11]
    }

    result = process_graph(node_dict, show_graph=True)
    print("Ordered list representation:", result)

    result = process_graph(aliased, show_graph=True)
    print("(Aliased) Ordered list representation:", result)
