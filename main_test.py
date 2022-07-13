import networkx as nx
import sys
sys.path.append("/Users/xuhaoshuai/GitHub/HumanIE/core/rule_match/graph")
from graph.words_of_graph import WordsOfGraph
from graph.shortest_path_graph import ShortestPathGraph
from k_score.k_counter import KCounter

if __name__ == '__main__':
    s1 = 'e1 e2 period'
    s2 = 'e1 e2'
    G1 = WordsOfGraph(s1, 2).get_graph()
    G2 = WordsOfGraph(s2, 2).get_graph()
    C1 = ShortestPathGraph(G1, 2).get_graph()
    C2 = ShortestPathGraph(G2, 2).get_graph()
    score = KCounter(C1, C2).get_k_score()
    print(score)
