import networkx as nx
from networkx.classes.graph import Graph
from words_of_graph import WordsOfGraph



class ShortestPathGraph:
    G = nx.Graph()
    d = int()
    C = nx.Graph()

    def __init__(self, words_of_graph:Graph, d=2):
        self.G = words_of_graph
        self.d = d
        self.C = self.construct_graph(self.G, self.d)
    
    def construct_graph(self, words_of_graph, d):
        path_length = dict(nx.all_pairs_dijkstra_path_length(words_of_graph))
        node_pair = list()
        for k, v in path_length.items():
            for m,n in v.items():
                if n <= d and n > 0:
                    node_pair.append([k, m, n])
        for i in node_pair:
                words_of_graph.add_edge(i[0], i[1], weight=1/i[2])
        return words_of_graph
    
    def get_graph(self):
        return self.C