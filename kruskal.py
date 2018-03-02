#!/bin/python3

import sys

parent = dict()
rank = dict()

def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0

def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]

def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
            if rank[root1] == rank[root2]: rank[root2] += 1

def mst(n, edges):
    overall_weight = 0
    for i in range(n):
        make_set(i)
    minimum_spanning_tree = set()
    for edge in edges:
        vertice1, vertice2, weight = edge
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.add((vertice1, vertice2))
            overall_weight += weight
    return overall_weight


if __name__ == "__main__":
    n, m = input().strip().split(' ')
    n, m = [int(n), int(m)]
    edges = []
    for edges_i in range(m):
        edges_t = [int(edges_temp)-1 for edges_temp in input().strip().split(' ')]
        edges_t[2] += 1
        edges.append(edges_t)
    edges.sort(key=lambda x: x[2])
    result = mst(n, edges)
    print(result)
