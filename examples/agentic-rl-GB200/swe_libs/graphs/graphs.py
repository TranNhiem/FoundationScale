"""Unweighted graph utilities.

Graphs are represented as adjacency lists: ``dict[Hashable, list[Hashable]]``.
All functions are side-effect free.
"""

from __future__ import annotations

from collections import deque


def bfs_order(graph: dict, start) -> list:
    """Return the breadth-first traversal order of ``graph`` from ``start``.

    Every node is visited exactly once; already-seen nodes are skipped.
    Raises ``KeyError`` if ``start`` is not present in ``graph``.
    """
    if start not in graph:
        raise KeyError(f"start node {start!r} not in graph")
    seen: set = {start}
    order: list = [start]
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbour in graph.get(node, []):
            if neighbour not in seen:
                seen.add(neighbour)
                order.append(neighbour)
                queue.append(neighbour)
    return order


def shortest_path_length(graph: dict, source, target) -> int | None:
    """Return the number of edges on the shortest ``source``->``target`` path.

    Returns ``None`` if ``target`` is unreachable from ``source``.
    Returns ``0`` when ``source == target``.
    """
    if source not in graph:
        raise KeyError(f"source node {source!r} not in graph")
    if source == target:
        return 0
    seen: set = {source}
    queue = deque([(source, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbour in graph.get(node, []):
            if neighbour == target:
                return dist + 1
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, dist + 1))
    return None


def connected_components(graph: dict) -> list:
    """Return the connected components of the undirected ``graph``.

    Each component is a sorted list of nodes. ``graph`` is assumed symmetric
    (undirected). Isolated nodes (with empty neighbour lists) form singleton
    components. Result components are sorted by their smallest element.
    """
    seen: set = set()
    components: list = []
    for node in sorted(graph, key=repr):
        if node in seen:
            continue
        comp: list = []
        queue = deque([node])
        seen.add(node)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for neighbour in graph.get(cur, []):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(sorted(comp, key=repr))
    components.sort(key=lambda c: repr(c[0]))
    return components


def has_cycle(graph: dict) -> bool:
    """Return ``True`` if the undirected ``graph`` contains any cycle.

    Uses a DFS walk tracking parent edges. A cycle exists if, during the
    traversal, an edge leads back to a visited node that is not the current
    node's parent tree edge.
    """
    visited: set = set()
    for root in sorted(graph, key=repr):
        if root in visited:
            continue
        # iterative DFS: (node, parent)
        stack: list = [(root, None)]
        visited.add(root)
        while stack:
            node, parent = stack.pop()
            for neighbour in graph.get(node, []):
                if neighbour == parent:
                    continue
                if neighbour in visited:
                    return True
                visited.add(neighbour)
                stack.append((neighbour, node))
    return False


def degree_sequence(graph: dict) -> list:
    """Return the list of node degrees sorted in descending order."""
    degrees = [len(graph.get(node, [])) for node in graph]
    degrees.sort(reverse=True)
    return degrees


def is_tree(graph: dict) -> bool:
    """Return ``True`` if ``graph`` is a tree (connected and acyclic)."""
    if not graph:
        return False
    if has_cycle(graph):
        return False
    return len(connected_components(graph)) == 1
