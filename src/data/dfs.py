import random
from collections import defaultdict


def dfs_edges(G, source, randomize_neighbors=False):
    nodes = [source]
    visited = set()

    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        successors = list(G[start])

        if randomize_neighbors:
            random.shuffle(successors)

        stack = [(start, iter(successors))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent, child
                    visited.add(child)

                    successors = list(G[child])
                    if randomize_neighbors:
                        random.shuffle(successors)

                    stack.append((child, iter(successors)))
            except StopIteration:
                stack.pop()


def dfs_successors(G, source, randomize_neighbors=False):
    d = defaultdict(list)
    for s, t in dfs_edges(G, source=source, randomize_neighbors=randomize_neighbors):
        d[s].append(t)
    return dict(d)
