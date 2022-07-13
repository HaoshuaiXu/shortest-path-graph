import networkx as nx
import nltk

class WordsOfGraph:
    text = str()
    token = list()
    window_size = int()
    wog = nx.Graph()

    def __init__(self, text, window_size=2):
        self.text =text
        self.set_token()
        self.window_size = window_size
        self.wog = self.construct_graph(self.token, self.window_size)
    
    def set_token(self):
        self.token = [w for w in nltk.word_tokenize(self.text)]
    
    def construct_graph(self,word_list, window_size):
        G = nx.Graph()
        for w in word_list:
            G.add_node(w)
            G.nodes[w]['label'] = w
        for i in range(len(word_list)):
            for j in range(i + 1, i + window_size):
                if j <len(word_list):
                    G.add_edge(word_list[i], word_list[j])
        return G
    
    def get_graph(self):
        return self.wog