import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
from networkx.algorithms.shortest_paths import weighted
from networkx.readwrite import text
from nltk.corpus import stopwords
from numpy import add, short
from transformers import BertTokenizer


class ShortestPath:
    s1 = str()
    s2 = str()
    window_size = int()
    d = int()
    threshold = int()
    token1 = list()
    token2 = list()
    G1 = nx.Graph()
    G2 = nx.Graph()
    C1 = nx.Graph()
    C2 = nx.Graph()
    score = int()
    tokenizer = ''

    def __init__(self, s1, s2, threshold:int, tokenizer:BertTokenizer, window_size=2, d=2) -> None:
        self.s1 = s1
        self.s2 = s2
        self.window_size = window_size
        self.d = d
        self.threshold = threshold
        self.tokenizer = tokenizer
        # self.text_preprocess()
        self.set_token()
        self.set_G1_G2()
        self.set_C1_C2()
        self.set_score()
    
    # def text_preprocess(self):
        # stpw = stopwords.words('english')
        # if self.my_stpw != None:
        #     for m in self.my_stpw:
        #         stpw.append(m)
        # self.token1 = [w for w in nltk.word_tokenize(s1) if w not in stpw]
        # self.token2 = [w for w in nltk.word_tokenize(s2) if w not in stpw]

    def set_token(self):
        self.token1 = [w for w in nltk.word_tokenize(self.s1)]
        self.token2 = [w for w in nltk.word_tokenize(self.s2)]

    def words_of_graph(self,word_list, window_size):
        G = nx.Graph()
        for w in word_list:
            G.add_node(w)
            G.nodes[w]['label'] = w
        for i in range(len(word_list)):
            for j in range(i + 1, i + window_size):
                if j <len(word_list):
                    G.add_edge(word_list[i], word_list[j])
        return G
    
    def set_G1_G2(self):
        self.G1 = self.words_of_graph(self.token1, self.window_size)
        self.G2 = self.words_of_graph(self.token2, self.window_size)

    def shortest_path_graph(self, words_of_graph, d):
        path_length = dict(nx.all_pairs_dijkstra_path_length(words_of_graph))
        node_pair = list()
        for k, v in path_length.items():
            for m,n in v.items():
                if n <= d and n > 0:
                    node_pair.append([k, m, n])
        for i in node_pair:
                words_of_graph.add_edge(i[0], i[1], weight=1/i[2])
        return words_of_graph
    
    def set_C1_C2(self):
        self.C1 = self.shortest_path_graph(self.G1, self.d)
        print(self.C1)
        self.C2 = self.shortest_path_graph(self.G2, self.d)
    
    def get_word_vec(self, word):
        word_vec = tokenizer.encode(
            text=word,
            add_special_tokens=False,
            padding='max_length',
            max_length=2,
            truncation=True,
            return_tensors='np'
            )
        return word_vec

    def k_node(self, v1, v2):
        v1_vec = self.get_word_vec(v1)[0]  # 外面总是包了一层[]，用[0]给去掉
        v2_vec = self.get_word_vec(v2)[0]
        cos_score = v1_vec.dot(v2_vec.T) / (np.linalg.norm(v1_vec) * np.linalg.norm(v2_vec))
        if cos_score >= self.threshold:
            return cos_score
        else:
            return int(0)

    def k_node_sum(self, C1, C2):
        V1 = C1.nodes
        V2 = C2.nodes
        k_node_sum = 0
        for v1 in V1:
            for v2 in V2:
                k_node = self.k_node(v1, v2)
                k_node_sum += k_node
        return k_node_sum
    
    def k_edge(self, e1, e2):
        l1 = e1['weight']
        l2 = e2['weight']
        k_edge = l1 * l2 * 2
        return k_edge

    def k_walk_sum(self, C1, C2):
        E1 = C1.edges
        E2 = C2.edges
        k_walk_sum = 0
        for e1 in E1:
            for e2 in E2:
                k_node_u = self.k_node(e1[0], e2[0])
                k_node_v = self.k_node(e1[1], e2[1])
                k_walk =  k_node_u * self.k_edge(E1[e1], E2[e2]) * k_node_v
                k_walk_sum += k_walk
        return k_walk_sum

    def get_norm(self, C1, C2):
        A1 = nx.to_numpy_matrix(C1)
        A2 = nx.to_numpy_matrix(C2)
        D1 = np.eye(N=A1.shape[0], M=A1.shape[1])
        D2 = np.eye(N=A2.shape[0], M=A2.shape[1])
        M1 = A1 + D1
        M2 = A2 + D2
        norm = np.linalg.norm(M1) * np.linalg.norm(M2)
        return norm

    def get_k_score(self,k_node, k_walk, norm):
        k_score = (k_node + k_walk) / norm
        return k_score
    
    def set_score(self):
        self.score = self.get_k_score(
            k_node=self.k_node_sum(self.C1, self.C2),
            k_walk=self.k_walk_sum(self.C1, self.C2),
            norm=self.get_norm(self.C1, self.C2))
    
    def get_score(self):
        return self.score


if __name__ == '__main__':
    sentence1 = "e1 e2 period"
    sentence2 = "e1 e2 period use"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test = ShortestPath(
        s1=sentence1,
        s2=sentence2,
        tokenizer=tokenizer,
        threshold=0.999,
        window_size=2,
        d=2
    )
    score = test.get_score()
    print(score)
    # my_stopwords = ['the', '.']
    # windows_size = 2
    # d = 2

    # s1 = text_preprocess(sentence1, my_stpw=my_stopwords)
    # s2 = text_preprocess(sentence2, my_stpw=my_stopwords)
    # G1 = words_of_graph(s1, windows_size)
    # G2 = words_of_graph(s2, windows_size)

    # print(nx.to_numpy_matrix(G1))

    # C1 = shortest_path_graph(G1, d)
    # C2 = shortest_path_graph(G2, d)
    # k_node = get_k_node(C1, C2)
    # k_walk = get_k_walk(C1, C2)
    # norm = get_norm(C1, C2)
    # print('knode = ' + str(k_node))
    # print('kwalk = ' + str(k_walk))
    # print('norm = ' + str(norm))
    # K = get_k_score(
    #     k_node=k_node,
    #     k_walk=k_walk,
    #     norm=norm
    # )
    # print(K)
