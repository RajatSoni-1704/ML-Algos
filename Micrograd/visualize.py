from collections import deque
from graphviz import Digraph

def track_tree(root_node):
    """
    Performs a Breadth-First Search (BFS) starting from the root_node
    to discover all nodes and edges in the computational graph.
    
    Returns:
        nodes (set): Set of all Value nodes in the graph.
        edges (set): Set of tuples representing directed edges (child, parent) in the graph.
    """
    nodes, edges = set(), set()
    visited = set()
    q = deque([root_node])
    
    while q:
        n = q.popleft()
        
        if n in visited:
            continue
        
        visited.add(n)
        nodes.add(n)
        
        # Traverse parent operands that created this node
        for j in list(n.prev):
            nodes.add(j)
            edges.add((n, j))
            q.append(j)
        
    return nodes, edges

def draw_graph(root_node):
    """
    Generates a Graphviz Digraph illustrating the computational graph.
    Each Value node displays its label (if any), value, and gradient.
    Operation nodes are displayed separately to show how values are combined.
    """
    nodes, edges = track_tree(root_node)
    
    # Left-to-Right orientation for the network visualization
    dot = Digraph(format="pdf", graph_attr={"rankdir": "LR"})
    
    for n in nodes:
        # Create a record-shape node for each Value object
        dot.node(str(id(n)), label=f"{n.label} | Data:{n.data:0.4f} | Grad:{n.grad:0.4f}", shape="record")
        
        # If this Value is the result of an operation, create an op node and link it
        if n.op:
            dot.node(str(id(n)) + n.op, label=f"{n.op}")
            dot.edge(str(id(n)) + n.op, str(id(n)))

    # Connect nodes based on mathematical dependencies
    for child, parent in edges:
        dot.edge(str(id(parent)), str(id(child)) + child.op)

    return dot

