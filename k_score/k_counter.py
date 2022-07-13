import networkx as nx
from networkx.classes.graph import Graph
import numpy as np

class KCounter:
    C1 = nx.Graph()
    C2 = nx.Graph()
    k_node_sum = int()
    k_walk_sum = int()
    norm = int()
    k_score = int()

    def __init__(self, C1:Graph, C2:Graph):
        self.C1 = C1
        self.C2 = C2
        self.k_node_sum = self.count_k_node_sum(self.C1, self.C2)
        self.k_walk_sum = self.count_k_walk_sum(self.C1, self.C2)
        self.norm = self.count_norm(self.C1, self.C2)
        self.k_score = self.count_k_score(self.k_node_sum, self.k_walk_sum, self.norm)

    def count_k_node_sum(self, C1:Graph, C2:Graph):
        V1 = C1.nodes
        V2 = C2.nodes
        k_node = 0
        for v1 in V1:
            if v1 in V2:
                k_node += 1
        return k_node

    def count_k_walk_sum(self, C1:Graph, C2:Graph):
        E1 = C1.edges
        E2 = C2.edges
        common_edge = list()
        k_walk = 0
        
        for e in E1:
            if e in E2:
                common_edge.append(e)
        
        for ce in common_edge:
            l1 = E1[ce]['weight']
            l2 = E2[ce]['weight']
            k_walk += l1 * l2 * 2
        return k_walk

    def count_norm(self, C1:Graph, C2:Graph):
        A1 = nx.to_numpy_matrix(C1)
        A2 = nx.to_numpy_matrix(C2)
        D1 = np.eye(N=A1.shape[0], M=A1.shape[1])
        D2 = np.eye(N=A2.shape[0], M=A2.shape[1])
        M1 = A1 + D1
        M2 = A2 + D2
        norm = np.linalg.norm(M1) * np.linalg.norm(M2)
        return norm

    def count_k_score(self,k_node, k_walk, norm):
        k_score = (k_node + k_walk) / norm
        return k_score
    
    def get_k_score(self):
        return self.k_score