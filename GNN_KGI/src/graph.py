import sys, random, os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datahelper import *
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import json

def Settings():
    with open('../model/settings.json') as f:
        settings = json.load(f)
    return settings

def negative_sampling(G, trainset):
    graphmes = GraphMes(G)
    nodecnt = graphmes.nodeCnt
    edgecnt = graphmes.edgeCnt
    h = np.random.randint(0, nodecnt, (trainset.shape[0],1))
    t = np.random.randint(0, nodecnt, (trainset.shape[0],1))
    r = trainset[:,1].reshape(-1,1)
    ns= np.concatenate((h,r,t), axis=1)
    # [i for i in trainset if i[1]!=graph.share_index+1]
    return ns

class nxClass(object):
    def __init__(self, root, sf):
        self.root = root
        self.G = self.readGraph(sf)
        
    def neighbors(self,node):
        return list(self.G.successors(node))+ list(self.G.predecessors(node)) + [node]

    def readGraph(self, sf, ints=False):
        helper = DataHelper(sf,NP=False)
        samples = helper.GetSamples()
        G  = nx.MultiDiGraph()
        if ints==False:
            for sample in samples:
                G.add_edge(sample[0],sample[2],attr=sample[1])
        elif ints==True:
            for sample in samples:
                G.add_edge(int(sample[0]),int(sample[2]),attr=int(sample[1]))
        return G

    @property
    def nodes(self):
        return self.G.nodes()

    def writeGraph(self, graph, f_o):
        with open(f_o,'w') as f:
            for edge in graph.edges():
                attr = graph[edge[0]][edge[1]][0]['attr']
                f.write(str(edge[0])+' '+str(attr)+' '+str(edge[1])+'\n')

    def divide(self, overlap_rate):
        if not os.path.isdir(self.root): os.mkdir(self.root)
        if not os.path.isdir(self.root+'/subG1'): os.mkdir(self.root+'/subG1')
        if not os.path.isdir(self.root+'/subG2'): os.mkdir(self.root+'/subG2')
        
        overlap_rate = 0.5*overlap_rate/(1-0.5*overlap_rate)
        node_size = len(self.nodes)
        nodes = np.array(self.nodes())
        np.random.shuffle(nodes)
        share_nodes = list(nodes[0:int(node_size*overlap_rate)])
        subg1_particular = nodes[int(node_size*overlap_rate)+1:int(0.5*node_size+0.5*node_size*overlap_rate)]
        subg2_particular = nodes[int(0.5*node_size+0.5*node_size*overlap_rate)+1:]
        subG1_nodes = list(share_nodes) + list(subg1_particular)
        subG2_nodes = list(share_nodes) + list(subg2_particular)
        print("All-subG1_nodes:",len(subG1_nodes))
        print("All-subG2_nodes:",len(subG2_nodes))
        # Divide into two parts
        subG1 = self.G.subgraph(subG1_nodes)
        subG2 = self.G.subgraph(subG2_nodes)
        # Wirte information of two graphs into folders
        self.share_nodes = set()
        subG1_nodes = set()
        subG2_nodes = set()
        with open(self.root + '/subG1/train.txt','w') as f:
            for edge in subG1.edges():
                attr = subG1[edge[0]][edge[1]][0]['attr']
                f.write(edge[0]+'_1 '+attr+'_1 '+edge[1]+'_1\n')
                subG1_nodes.add(edge[0])
                subG1_nodes.add(edge[1])
        with open(self.root+'/subG2/train.txt','w') as f:
            for edge in subG2.edges():
                attr = subG2[edge[0]][edge[1]][0]['attr']
                f.write(edge[0]+'_2 '+attr+'_2 '+edge[1]+'_2\n')
                subG2_nodes.add(edge[0])
                subG2_nodes.add(edge[1])
        
        for node in subG2_nodes | subG1_nodes:
            if node in subG2_nodes and node in subG1_nodes:
                self.share_nodes.add(node)
        print("After-share_part:",len(share_nodes))
        print("After-subG1_nodes:",len(subG1_nodes))
        print("After-subG2_nodes:",len(subG2_nodes))

    def integrate_text(self):
        self.GI = nx.MultiDiGraph()
        share_nodes = list(self.share_nodes)
        self.subG1 = self.readGraph(self.root+'/subG1/train.txt')
        self.subG2 = self.readGraph(self.root+'/subG2/train.txt')
        self.GI = nx.compose(self.subG1, self.subG2)
        previous_len = len(self.GI.edges())
        for i in self.share_nodes:
            self.GI.add_edge(i+'_1',i+'_2',attr="SHARE")
        assert(len(self.share_nodes)+previous_len==len(self.GI.edges()))
        return self.GI

    def integrate(self, train_rate):
        # New directory
        if not os.path.isdir(self.root+'/GI'): os.mkdir(self.root+'/GI')
        if not os.path.isdir(self.root+'/GI'+'/subGs'): os.mkdir(self.root+'/GI'+'/subGs')
        if not os.path.isdir(self.root+'/GI'+'/GI'): os.mkdir(self.root+'/GI'+'/GI')
        # Basic information of two subgraph
        self.GI = nx.MultiDiGraph()
        subG1_helper = DataHelper(self.root+'/subG1/train.txt')
        subg1_nodes_cnt = len(subG1_helper.nodes)
        subg1_edges_cnt = len(subG1_helper.edges)
        subG2_helper = DataHelper(self.root+'/subG2/train.txt', nodeIndexStart=subg1_nodes_cnt, edgeIndexStart=subg1_edges_cnt)
        # 把之前的图index化
        subG1_helper.id2file()
        subG2_helper.id2file()
        share_index = len(subG1_helper.edge2id) + len(subG2_helper.edge2id)
        subG1 = self.readGraph(self.root+'/subG1/train.txt')
        subG2 = self.readGraph(self.root+'/subG2/train.txt')
        self.rewrite(subG1, subG1_helper, self.root+'/subG1/train_idx.txt')
        self.rewrite(subG2, subG2_helper, self.root+'/subG2/train_idx.txt')
        self.subG1 = self.readGraph(self.root+'/subG1/train_idx.txt', ints=True)
        self.subG2 = self.readGraph(self.root+'/subG2/train_idx.txt', ints=True)
        # 合并两个index化之后的图
        self.GI = nx.compose(self.subG1, self.subG2)
        previous_len = len(self.GI.edges())
        share_nodes = list(self.share_nodes)
        share_nodes_train   = share_nodes[:int(train_rate*len(share_nodes))]
        share_nodes_test    = share_nodes[int(train_rate*len(share_nodes)):]
        assert(len(share_nodes_train)+len(share_nodes_test)==len(share_nodes))
        for i in share_nodes_train:
            self.GI.add_edge(subG1_helper.node2id[i+"_1"],subG2_helper.node2id[i+"_2"],attr=share_index)
            self.GI.add_edge(subG2_helper.node2id[i+"_2"],subG1_helper.node2id[i+"_1"],attr=share_index+1)
        assert(len(share_nodes_train)*2+previous_len==len(self.GI.edges()))
        self.writeGraph(self.GI, self.root+'/GI/GI/GI.txt')
        # 将 share_nodes_train, share_nodes_train 的信息写入文件.
        with open(self.root+'/GI/subGs/share_nodes_train.txt', 'w') as f:
            for i in share_nodes_train:
                f.write(str(subG1_helper.node2id[i+"_1"])+" ")
                f.write(str(subG2_helper.node2id[i+"_2"])+'\n')
        with open(self.root+'/GI/subGs/share_nodes_test.txt', 'w') as f:
            for i in share_nodes_test:
                f.write(str(subG1_helper.node2id[i+"_1"])+" ")
                f.write(str(subG2_helper.node2id[i+"_2"])+'\n')
        # 合并GI的id信息
        self.node2id, self.edge2id, self.id2node, self.id2edge = {}, {},{}, {}
        self.node2id.update(subG1_helper.node2id)
        self.node2id.update(subG2_helper.node2id)
        texthelper.dict2file(self.node2id, self.root+'/GI'+'/subGs/entity2id2.txt')
        self.edge2id.update(subG1_helper.edge2id)
        self.edge2id.update(subG2_helper.edge2id)
        texthelper.dict2file(self.edge2id, self.root+'/GI'+'/subGs/relation2id2.txt')
        self.share_index = share_index
        self.edge2id.update({"SHARE_1":share_index,"SHARE_2":share_index+1})
        self.id2node = {v: k for k, v in self.node2id.items()}
        self.id2edge = {v: k for k, v in self.edge2id.items()}
        texthelper.dict2file(self.node2id, self.root+'/GI'+'/subGs/entity2id.txt')
        texthelper.dict2file(self.edge2id, self.root+'/GI'+'/subGs/relation2id.txt')
        
        return self.GI
        
    # rewrite graph triples in form of index
    def rewrite(self, graph, data_helper, of):
        with open(of, 'w') as f:
            for edge in graph.edges():
                relation = str(data_helper.edge2id[graph[edge[0]][edge[1]][0]['attr']])
                head = str(data_helper.node2id[edge[0]])
                tail = str(data_helper.node2id[edge[1]])
                f.write(head+' '+relation+' '+tail+'\n')
    
    def dataset(self, train_rate):
        if not os.path.isdir(self.root+'/train'): os.mkdir(self.root+'/train')
        if not os.path.isdir(self.root+'/test'): os.mkdir(self.root+'/test')
        dataset = []
        for edge in self.GI.edges():
            attr = self.GI[edge[0]][edge[1]][0]['attr']
            dataset.append(np.array([edge[0],attr,edge[1]]))
        np.random.shuffle(np.array(dataset))
        train_index = int(train_rate*len(dataset))
        
        trainset = np.array(dataset[:train_index])
        testset = np.array(dataset[train_index:])
        testset_share   = np.array([i for i in testset if i[1]>=self.share_index])
        testset_common  = np.array([i for i in testset if i[1]<self.share_index])
        trainset = np.concatenate((trainset,testset_common), axis=0)
        
        return trainset, testset
    
class GraphMes:
    def __init__(self, graph=None, file=None, start=0):
        if graph==None and file!=None:
            self.G = self.readGraph(self, sf, ints=False)
        elif graph!=None and file==None:
            self.G = graph
        else:
            raise Exception
        self.start = start
        self._node2id()
        self._edge2id()

    def readGraph(self, sf, ints=False):
        helper = DataHelper(sf,NP=False)
        samples = helper.GetSamples()
        self.SamplesCnt = len(samples)
        G  = nx.MultiDiGraph()
        if ints==False:
            for sample in samples:
                G.add_edge(sample[0],sample[2],attr=sample[1])
        elif ints==True:
            for sample in samples:
                G.add_edge(int(sample[0]),int(sample[2]),attr=int(sample[1]))
        return G

    def neighbors_zx(self,node):
        return list(self.G.successors(node))+ list(self.G.predecessors(node)) + [node]
    
    def tensor(self):
        pre_indexs, predecessors, predecessors_group = [], [], []
        suc_indexs, successors, successors_group = [], [], []
        index = 0
        # print("sample cnt:", mes.samplesCnt)
        nodes = self.G.nodes()
        for node in nodes:
            j = 0
            for i in self.G.successors(node):
                attr = self.G[node][i][0]['attr']
                successors.append([i,attr])
                j+=1
            index = index+j
            suc_indexs.append(index)
        # 这里就是将扁平化的邻接信息按照node进行分割
        successors_group = np.split(successors,suc_indexs, axis=0)[:-1]
        
        index = 0   
        for node in nodes:
            j = 0
            for i in self.G.predecessors(node):
                attr = self.G[i][node][0]['attr']
                predecessors.append([i,attr])
                j+=1
            index = index+j
            pre_indexs.append(index)
        predecessors_group = np.split(predecessors,pre_indexs, axis=0)[:-1]
        self._check_index(predecessors_group, successors_group)
        # print("len(nodes):",len(nodes))
        # print("len(successors_group):",len(successors_group))
        # print("len(successors):",len(successors))
        self.predecessors_group = predecessors_group
        self.successors_group = successors_group
    
    # predecessors : 持有最初始的邻接信息, flat化的, 没有每个node的信息
    # predecessorss : 和predecessors持有一样的内容, 但是里面根据index进行了划分, 指出了每个node拥有的邻接单元
    # predecessorsss : 将 predecessors 中的node作为出发点, 再次找其下的邻接信息, 这里对应的是第二个圈.
    def tensor2(self, nodes):
        pre_indexs, predecessors = [], []
        suc_indexs, successors = [], []
        index = 0
        for node in nodes:
            j = 0
            for i in self.G.successors(node):
                attr = self.G[node][i][0]['attr']
                successors.append([i,attr])
                j+=1
            index = index+j
            suc_indexs.append(index)
        # 这里就是将扁平化的邻接信息按照node进行分割
        successors_group = np.split(successors,suc_indexs, axis=0)[:-1]
        
        index = 0   
        for node in nodes:
            j = 0
            for i in self.G.predecessors(node):
                attr = self.G[i][node][0]['attr']
                predecessors.append([i,attr])
                j+=1
            index = index+j
            pre_indexs.append(index)
        predecessors_group = np.split(predecessors,pre_indexs, axis=0)[:-1]
        suc_mes = Adjacency(successors, suc_indexs)
        pre_mes = Adjacency(predecessors, pre_indexs)
        self._check_index(predecessors_group, successors_group)

        suc_suc_indexss, suc_pre_indexss, pre_pre_indexss, pre_suc_indexss = [], [], [], []
        suc_successorss, suc_predecessorss, pre_successorss, pre_predecessorss = [], [], [], []
        for i in successors_group:
            for j in i:
                for k in self.successors_group[j[0]]:
                    suc_successorss.append(k)
                for k in self.predecessors_group[j[0]]:
                    suc_predecessorss.append(k)
                suc_suc_indexss.append(len(suc_successorss))
                suc_pre_indexss.append(len(suc_predecessorss))
        suc_successorss = np.array(suc_successorss)
        suc_predecessorss = np.array(suc_predecessorss)
        suc_pre_mes = Adjacency(suc_predecessorss, suc_pre_indexss)
        suc_suc_mes = Adjacency(suc_successorss, suc_suc_indexss)        
    
        for i in predecessors_group:
            for j in i:
                for k in self.successors_group[j[0]]:
                    pre_successorss.append(k)
                for k in self.predecessors_group[j[0]]:
                    pre_predecessorss.append(k)
                pre_suc_indexss.append(len(pre_successorss))
                pre_pre_indexss.append(len(pre_predecessorss))
        pre_successorss = np.array(pre_successorss)
        pre_predecessorss = np.array(pre_predecessorss)
        # print("pre_predecessorss:", pre_predecessorss)
        # print("pre_pre_indexss", pre_pre_indexss)
        pre_pre_mes = Adjacency(pre_predecessorss, pre_pre_indexss)
        # print("pre_successorss:", pre_successorss)
        # print("pre_suc_indexss", pre_suc_indexss)
        pre_suc_mes = Adjacency(pre_successorss, pre_suc_indexss)
        # self._check_index(pre_pre_mes.pair_group, pre_suc_mes.pair_group)
        # print("\nlen(nodes):",len(nodes))
        # print("len(suc_indexs)",len(suc_indexs))
        # print("len(successors):",len(successors))
        # print("len(suc_suc_indexss):",len(suc_suc_indexss))
        # print("len(suc_pre_indexss):",len(suc_pre_indexss))
        # print("len(suc_successorss):",len(suc_successorss))
        # print("len(suc_predecessorss):",len(suc_predecessorss))
        # print('====================================')
        # print("len(pre_indexs)",len(pre_indexs))
        # print("len(predecessor):",len(predecessors))
        # print("len(pre_suc_indexss):",len(pre_suc_indexss))
        # print("len(pre_pre_indexss):",len(pre_pre_indexss))
        # print("len(pre_successorss):",len(pre_successorss))
        # print("len(pre_predecessorss):",len(pre_predecessorss))
        
        return [[pre_mes,[pre_pre_mes, pre_suc_mes]], [suc_mes,[suc_pre_mes, suc_suc_mes]]]
    
    def _check_index(self, pre, suc):
        assert(len(pre)==len(suc))
        for i in range(len(pre)):
            assert(len(pre[i])>0 or len(suc[i])>0)
        
    def _node2id(self):
        self.node2id = dict()
        self.id2node = dict()
        index = 0
        for node in self.G.nodes():
            self.node2id.update({node:self.start+index})
            self.id2node.update({self.start+index:node})
            index += 1
    def _edge2id(self):
        self.edge2id = dict()
        self.id2edge = dict()
        self.attrs = set()
        for edge in self.G.edges():
            attr = self.G[edge[0]][edge[1]][0]['attr']
            self.attrs.add(attr)
        index = 0
        for attr in self.attrs:
            self.edge2id.update({attr:self.start+index})
            self.id2edge.update({self.start+index:attr})
            index += 1
    @property
    def nodes(self):
        return list(self.G.nodes)
    @property
    def nodeCnt(self):
        return len(self.G.nodes)
    @property
    def samplesCnt(self):
        return self.SamplesCnt
    @property
    def edges(self):
        return list(self.attrs)
    @property
    def edgeCnt(self):
        return len(self.attrs)
    def id2file(self, nodefn, edgefn):
        with open(nodefn, 'w') as nf:
            for i in range(len(self.node2id)):
                nf.write(str(self.id2node[i])+' '+str(i)+'\n')
        with open(edgefn, 'w') as ef:
            for i in range(len(self.edge2id)):
                ef.write(str(self.id2edge[i])+' '+str(i)+'\n')

class Adjacency:
    def __init__(self,pairs,indexs):
        self.pairs = np.array(pairs)
        self.indexs = indexs
        self.pair_group = np.split(pairs,indexs, axis=0)[:-1]
        if len(pairs)!=0:
            self.nodes  = np.array(self.pairs[:,0])
            self.links  = np.array(self.pairs[:,1])
        else:
            self.nodes  = np.array([])
            self.links  = np.array([])

    
if __name__ == "__main__":
    settings = Settings()    

    def main(root, overlap_rate, train_rate):
        sf = "../data/train_data/"+root+".txt"
        sf_sign = sf.split('/')[-1].split('.')[0]+'_'
        root = './../data/'+sf_sign+"O"+str(int(overlap_rate*100))+'T' +str(int(train_rate*100))
        
        graph = nxClass(root, sf)
        overlap_rates = settings['overlap_rate'][0]
        train_rates = settings['train_rate'][0]
        # print(overlap_rates)
        # construct graph
        graph.divide(overlap_rates)
        GI = graph.integrate(train_rates)
        mes = GraphMes(GI)
        mes.id2file(root+"/GI/GI/entity2id.txt",root+"/GI/GI/relation2id.txt")

    root = ['FBL','FBB','FBM','FBS','WNB','WNM','WNS','WNMI']
    # root = ['FBS']
    for i in root:
        for overlap_rate in settings['overlap_rate']:
            for train_rate in settings['train_rate']:
                main(i, overlap_rate, train_rate)

