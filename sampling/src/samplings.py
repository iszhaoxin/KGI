import matplotlib.pyplot as plt
import os, re, json, math, copy
import mylib.texthelper.decorator as decorator
import mylib.texthelper.format as texthelper
import numpy as np
import itertools
import networkx as nx
from graph import GraphMes
from scipy.sparse import csr_matrix

class Sampling:
    def __init__(self, root, graphmes, overlapRate, trainRate):
        self.G              = graphmes.G
        self._root           = root
        self.uG             = nx.Graph(graphmes.G)
        self.graphMes       = graphmes
        self.overlapRate    = overlapRate
        self.trainRate      = trainRate
        self.nodes          = np.array(graphmes.nodes)
        self.node2id, self.id2node = graphmes.node2id, graphmes.id2node
        self.edge2id, self.id2edge = graphmes.edge2id, graphmes.id2edge

    def _record(self, statistic=False):
        # Creat file
        if not os.path.isdir(self.root+'/subG1'): os.mkdir(self.root+'/subG1')
        if not os.path.isdir(self.root+'/subG2'): os.mkdir(self.root+'/subG2')
        subG1 = self.G.subgraph(self.subG1_nodes)
        subG2 = self.G.subgraph(self.subG2_nodes)
        
        with open(self.root + '/subG1/train.txt','w') as f:
            for edge in subG1.edges():
                attr = subG1[edge[0]][edge[1]][0]['attr']
                f.write(str(edge[0])+' '+str(attr)+' '+str(edge[1])+'\n')
        with open(self.root+'/subG2/train.txt','w') as f:
            for edge in subG2.edges():
                attr = subG2[edge[0]][edge[1]][0]['attr']
                f.write(str(edge[0])+' '+str(attr)+' '+str(edge[1])+'\n')
        
        graphmes1 = GraphMes(logging=self.root+'/subG1', file=self.root+'/subG1/train.txt', ints=True)
        graphmes2 = GraphMes(logging=self.root+'/subG2', file=self.root+'/subG2/train.txt', ints=True)

        self.subG1_nodes = set(graphmes1.nodes)
        self.subG2_nodes = set(graphmes2.nodes)
        self.share_nodes = set(self.share_nodes)
        print('len(self.share_nodes):', len(self.share_nodes))
        self.share_nodes = self.share_nodes & self.subG1_nodes & self.subG2_nodes
        print('len(self.share_nodes)_after:', len(self.share_nodes))


        
        print('graphmes1 zipf begin -----')
        graphmes1.zipf(plot=True)
        print('graphmes1.zipf over  +++++')
        
        print('graphmes1.zipf_coeffi begin -----')
        graphmes1.zipf_coeffi(plot=True)
        print('graphmes1.zipf_coeffi over  +++++')
        
        print('graphmes2.zipf begin -----')
        graphmes2.zipf(plot=True)
        print('graphmes2.zipf over  +++++')
        
        print('graphmes2.zipf_coeffi begin -----')
        graphmes2.zipf_coeffi(plot=True)
        print('graphmes2.zipf_coeffi over  +++++')

        if statistic == True:
            print('graphmes1 graphmes2 GraphMes begin -----')
            print('graphmes1 graphmes2 GraphMes over  +++++')
            print('graphmes1 record begin -----')
            graphmes1.record(additional=True)
            print('graphmes1 record over  +++++')
            
            
            print('graphmes2.record begin -----')
            graphmes2.record(additional=True)
            print('graphmes2.record over  +++++')
            
    def dataset(self):
        if not os.path.isdir(self.root+'/train'): os.mkdir(self.root+'/train')
        if not os.path.isdir(self.root+'/test'): os.mkdir(self.root+'/test')
        
        self.share_nodes = np.array(list(self.share_nodes))
        self.subG1_nodes = np.array(list(self.subG1_nodes))
        self.subG2_nodes = np.array(list(self.subG2_nodes))
        # Divide into trainset and testset
        np.random.shuffle(self.share_nodes)
        share_train, share_test = self.share_nodes[0:int(self.trainRate*len(self.share_nodes))], self.share_nodes[int(self.trainRate*len(self.share_nodes)):]
        assert(len(share_train)+len(share_test)==len(self.share_nodes))
        np.random.shuffle(self.subG1_nodes)
        subg1_train, subg1_test = self.subG1_nodes[0:int(self.trainRate*len(self.subG1_nodes))], self.subG1_nodes[int(self.trainRate*len(self.subG1_nodes)):]
        assert(len(subg1_train)+len(subg1_test)==len(self.subG1_nodes))
        np.random.shuffle(self.subG2_nodes)
        subg2_train, subg2_test = self.subG2_nodes[0:int(self.trainRate*len(self.subG2_nodes))], self.subG2_nodes[int(self.trainRate*len(self.subG2_nodes)):]
        assert(len(subg2_train)+len(subg2_test)==len(self.subG2_nodes))
        # Write into files
        with open(self.root+'/train/share_train.txt', 'w') as f:
            for i in share_train:
                f.write(str(i)+'\n')
        with open(self.root+'/train/subg1_train.txt', 'w') as f:
            for i in subg1_train:
                f.write(str(i)+'\n')
        with open(self.root+'/train/subg2_train.txt', 'w') as f:
            for i in subg2_train:
                f.write(str(i)+'\n')
        with open(self.root+'/test/share_test.txt', 'w') as f:
            for i in share_test:
                f.write(str(i)+'\n')
        with open(self.root+'/test/subg1_test.txt', 'w') as f:
            for i in subg1_test:
                f.write(str(i)+'\n')
        with open(self.root+'/test/subg2_test.txt', 'w') as f:
            for i in subg2_test:
                f.write(str(i)+'\n')
        
    def Uniform(self,record=True):
        self.root = self._root+'/uniform'
        if not os.path.isdir(self.root): os.mkdir(self.root)
        np.random.shuffle(self.nodes)
        
        self.overlapRate = 0.5*self.overlapRate/(1-0.5*self.overlapRate)
        node_size = len(self.nodes)
        self.share_nodes = list(self.nodes[0:int(node_size*self.overlapRate)])
        subg1_particular = self.nodes[int(node_size*self.overlapRate)+1:int(0.5*node_size+0.5*node_size*self.overlapRate)]
        subg2_particular = self.nodes[int(0.5*node_size+0.5*node_size*self.overlapRate)+1:]
        self.subG1_nodes = list(self.share_nodes) + list(subg1_particular)
        self.subG2_nodes = list(self.share_nodes) + list(subg2_particular)
        
        if record == True:
            print('Uniform _record begin -----')
            self._record(statistic=False)
            print('Uniform _record over +++++')
            
            print('Uniform dataset begin -----')
            self.dataset()
            print('Uniform dataset over  -----')
    
    def Weighted(self, record=True, plan=1):    
        degree = {}
        nodes = np.array(self.nodes)
        np.random.shuffle(nodes)
        for i in nodes:
            if self.G.degree(i) in degree:
                degree[self.G.degree(i)].add(i) 
            else:
                degree.update({self.G.degree(i):{i}})
        degree_percent = {}
        
        if plan==1:
            self.root = self._root+'/weighted_i*i'
            if not os.path.isdir(self.root): os.mkdir(self.root)
            _sum = sum([i*i for i in degree])
            _max = max([i*i for i in degree])
            rate = _sum/_max
            for i in degree:
                degree_percent.update({i:i*i/_sum*rate})
            print(degree_percent)
            print(max(list(degree_percent.values())))
        elif plan==2:
            self.root = self._root+'/weighted_i'
            if not os.path.isdir(self.root): os.mkdir(self.root)
            _sum = sum([i for i in degree])
            for i in degree:
                degree_percent.update({i:i/_sum})
        elif plan==3:
            self.root = self._root+'/weighted_i*log(i)'
            if not os.path.isdir(self.root): os.mkdir(self.root)
            _sum = sum([i*math.log(i) for i in degree])
            for i in degree:
                degree_percent.update({i:i*math.log(i)/_sum})

        # Fill share_nodes
        self.share_nodes = []
        node_size = len(self.nodes)
        nodes = set(self.nodes)
        while(len(self.share_nodes)<int(node_size*self.overlapRate)):
            selected_nodes = set()
            for i in nodes:
                prob = degree_percent[self.G.degree(i)]
                if np.random.rand() <= prob:
                    self.share_nodes.append(i)
                    selected_nodes.add(i)
                    if len(self.share_nodes)==int(node_size*self.overlapRate):
                        break
            for j in selected_nodes:
                nodes.remove(j)
        
        # Fill sub-graph nodes set
        self.subG1_nodes, self.subG2_nodes = [], []
        for i in degree:
            l = int(len(degree[i])/2)
            for j in range(l):
                self.subG1_nodes.append(degree[i].pop())
                self.subG2_nodes.append(degree[i].pop())
            while(len(degree[i]) != 0):
                if np.random.rand() <= 0.5:
                    self.subG1_nodes.append(degree[i].pop())
                else:
                    self.subG2_nodes.append(degree[i].pop())
        
        if record == True:
            print('Weighted _record begin -----')
            self._record(statistic=False)
            print('Weighted _record over +++++')
            
            print('dataset dataset begin -----')
            # self.dataset()
            print('dataset dataset over +++++')
    
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
            print("???????????????")
            random_sampling = np.random.randint(0, len(self.nodes)-1)
            while( random_sampling not in searched and len(margin)==0):
                margin.add(random_sampling)
                random_sampling = np.random.randint(0, len(self.nodes)-1)
        
    def Cohesive(self, record=True):
        self.root = self._root+ '/cohesive'
        if not os.path.isdir(self.root): os.mkdir(self.root)
        
        node_size = len(self.nodes)
        share_node_size = int(node_size*self.overlapRate)
        self.share_nodes = set()
        
        # all_bs = nx.betweenness_centrality(self.G)
        # all_bs = nx.eigenvector_centrality(self.uG)
        all_bs = nx.degree_centrality(self.uG)
        all_bs_sort = texthelper.sortDict(all_bs, By="value", reverse=True)
        # print(all_bs_sort)
        searched = set()
        margin = set()
        self.share_nodes = set()
        print(all_bs_sort[0:10])
        center = all_bs_sort[0][0]

        self.share_nodes.add(center)
        margin.add(center)
        searched.add(center)

        while(len(self.share_nodes) < share_node_size):
            print('++')
            margin_bs = {}
            self._update_margin(searched, margin)
            for i in margin:
                margin_bs.update({i:all_bs[i]})
            margin_bs_sort = texthelper.sortDict(margin_bs, By="value", reverse=True)
            # for i in margin_bs_sort:
            #     print(i)
            # print('-----------------------')
            for j in margin_bs_sort:
                self.share_nodes.add(j[0])
                if len(self.share_nodes) >= share_node_size:
                    break
        
        self.subG1_nodes, self.subG2_nodes = [], []
        residue_nodes = np.array(list(set(self.nodes)-self.share_nodes))
        np.random.shuffle(residue_nodes)
        residue_nodes = set(residue_nodes)
        
        for i in range(int(len(residue_nodes)/2)):
            self.subG1_nodes.append(residue_nodes.pop())
            self.subG2_nodes.append(residue_nodes.pop())
        while(len(residue_nodes) != 0):
            if np.random.rand() <= 0.5:
                self.subG1_nodes.append(residue_nodes.pop())
            else:
                self.subG2_nodes.append(residue_nodes.pop())
        
        if record == True:
            print('cohesive _record begin -----')
            self._record(statistic=False)
            print('cohesive _record over +++++')
            
            print('cohesive dataset begin -----')
            self.dataset()
            print('cohesive dataset over +++++')
    
    def test(self):
        degree = {}
        nodes = np.array(self.nodes)
        np.random.shuffle(nodes)

        max_degree = 0
        for i in nodes:
            if self.G.degree(i) > max_degree:
                max_degree = self.G.degree(i)
            if self.G.degree(i) in degree:
                degree[self.G.degree(i)].add(i) 
            else:
                degree.update({self.G.degree(i):{i}})
        
        for i in degree:
            degree_cnt = np.zeros(max_degree, dtype=int)
            print('------------------')
            for j in degree[i]:
                degree_of_j = self.G.degree(j)
                # print(degree_of_j)
                degree_cnt[degree_of_j] += 1
            # self.plot(i, degree_cnt)
    
    def plot(self, i, degree_cnt):
        plt.figure()
        xdata = np.array(range(len(degree_cnt)))
        ydata = degree_cnt
        plt.plot(xdata,ydata,'.')
        plt.savefig(self._root+"/"+str(i)+".png")
        plt.close(0)

if __name__ == "__main__":
    # graphfn = '../../data/fb15k_small/WNS_id.txt'
    graphfn = '../../data/fb15k/train_id.txt'
    # graphfn = '../../data/wordnet/WNB_id.txt'   
    
    root ="../results/"+ graphfn.split('/')[-2]+'/50_70_4'
    if not os.path.isdir(root): os.mkdir(root)
    
    # Data analysis of raw graph
    graphmes = GraphMes(logging=root+'/rawGraph', file=graphfn, ints=True)
    # graphmes.record()
    # graphmes.zipf(plot=True)
    # graphmes.zipf_coeffi(plot=True)
    
    # Sampling sub-graph
    sampling = Sampling(root, graphmes, overlapRate=0.3, trainRate=0.7)

    sampling.Uniform(record=True)
    # sampling.Weighted(record=True, plan=1)
    # sampling.Weighted(record=True, plan=2)
    # sampling.Weighted(record=True, plan=3)
    # sampling.Cohesive(record=True)

    # sampling.test()