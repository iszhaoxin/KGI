import numpy as np
import sys,traceback,os
import networkx as nx
import matplotlib.pyplot as plt
from datahelper import DataHelper
from scipy import stats,optimize
import mylib.texthelper.format as texthelper
import copy

class GraphMes:
    def __init__(self, graph=None, file=None, start=0, ints=False):        
        if graph==None and file!=None:  
            self.helper = DataHelper(file,NP=False)
            self.samples = self.helper.GetSamples()
            if ints == True:
                samples = []
                for i in self.samples:
                    samples.append([int(i[0]), int(i[1]),int(i[2])])
            self.samples = np.array(samples)
            self.G = self.readGraph(file, ints=ints)
            self.uG = self.readGraph(file, ints=ints, unweight = True)
        
        elif graph!=None and file==None:
            self.G = graph
            self.uG = nx.Graph(self.G)

            self.samples = []
            for edge in self.G.edges():
                for i in self.G[edge[0]][edge[1]]:
                    self.samples.append([edge[0], self.G[edge[0]][edge[1]][i]['attr'], edge[1]])

        else:
            raise Exception
        
        self.start = start
        self.node2id, self.id2node = self._node2id()
        self.edge2id, self.id2edge = self._edge2id()

    def readGraph(self, sf, ints=False, unweight = False):
        self.SamplesCnt = len(self.samples)
        if unweight == True:
            G  = nx.Graph()
            for sample in self.samples:
                G.add_edge(sample[0],sample[2])
        else:
            G  = nx.MultiDiGraph()
            for sample in self.samples:
                G.add_edge(sample[0],sample[2],attr=sample[1])
        return G

    def graph2id(self, of):    
        with open(of, 'w') as f:
            for h,r,t in self.samples:
                f.write(str(self.node2id[h])+' '+str(self.edge2id[r])+' '+str(self.node2id[t])+'\n')

    def _node2id(self):
        node2id = dict()
        id2node = dict()
        index = 0
        for node in self.G.nodes():
            node2id.update({node:self.start+index})
            id2node.update({self.start+index:node})
            index += 1
        return node2id, id2node

    def _edge2id(self):
        edge2id = dict()
        id2edge = dict()
        self.attrs = set()
        for edge in self.G.edges():
            for i in self.G[edge[0]][edge[1]]:
                # print(self.G[edge[0]][edge[1]][i]['attr'])
                self.attrs.add(self.G[edge[0]][edge[1]][i]['attr'])
        index = 0
        for attr in self.attrs:
            edge2id.update({attr:self.start+index})
            id2edge.update({self.start+index:attr})
            index += 1
        return edge2id, id2edge
    
    def id2file(self, nodefn, edgefn):
        with open(nodefn, 'w') as nf:
            for i in range(len(self.node2id)):
                nf.write(self.id2node[i]+' '+str(i)+'\n')
        with open(edgefn, 'w') as ef:
            for i in range(len(self.edge2id)):
                ef.write(self.id2edge[i]+' '+str(i)+'\n')

    def _update_margin(self, searched, margin):
        margin_backup = copy.copy(margin)
        for i in margin_backup:
            for j in self.G.neighbors(i):
                if j not in searched:
                    margin.add(j)
        for i in margin_backup:
            margin.remove(i)
            searched.add(i)
            
        if len(margin) == 0:
            random_sampling = np.random.randint(0, len(self.nodes)-1)
            while( random_sampling not in searched and len(margin)==0):
                margin.add(random_sampling)
                random_sampling = np.random.randint(0, len(self.nodes)-1)
        
    
    def cohesive(self, windowSize):
        all_bs = nx.eigenvector_centrality(self.uG)

        searched = set()
        margin = set()
        windows = set()
        center = np.random.randint(0,len(self.nodes)-1)
        windows.add(center)
        margin.add(center)
        searched.add(center)

        while(len(windows) < windowSize):
            margin_bs = {}
            self._update_margin(searched, margin)
            for i in margin:
                margin_bs.update({i:all_bs[i]})
            margin_bs_sort = texthelper.sortDict(margin_bs, By="value", reverse=True)
            for j in margin_bs_sort:
                windows.add(j[0])
                if len(windows) >= windowSize:
                    break
        return windows
        
    @property
    def nodes(self):
        return list(self.G.nodes)
    @property
    def nodeCnt(self):
        return len(self.G.nodes)
    @property
    def samplesCnt(self):
        return len(self.samples)
    @property
    def edges(self):
        return list(self.attrs)
    @property
    def edgeCnt(self):
        return len(self.attrs)
    

if __name__ == "__main__":
    # graphfn = '../../data/fb15k_small/train_id.txt'
    graphfn = '../../data/fb15k/train_id.txt'
    # graphfn = '../../data/wordnet/WNB_id.txt'   
    graphmes = GraphMes(file=graphfn, ints=True)
    cohesive = graphmes.cohesive(16)
    print(cohesive)