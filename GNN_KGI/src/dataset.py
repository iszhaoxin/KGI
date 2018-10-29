import os, inspect
import numpy as np
import networkx as nx
from datahelper import *
from chainer import Variable
# from mylib.texthelper.format import pshape, pdata, ptree

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def print_withname(variable):
    variable_name = retrieve_name(variable)
    print(variable_name+'\'s type: '+type(variable))

def pshape(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    if isinstance(var, np.ndarray):
        print(var_names[0]+".shape:", var.shape)
    elif isinstance(var, Variable):
        print(var_names[0]+".shape:", var.shape)    
    else:
        print(var_names[0]+".shape:", len(var))
def pdata(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    if isinstance(var, np.ndarray):
        if len(var.shape)>1:
            print(var_names[0]+".data:", var[0])
        else:
            print(var_names[0]+".data:", var)
    elif isinstance(var, Variable):
        if len(var.shape)>1:
            print(var_names[0]+".data:", var[0])
        else:
            print(var_names[0]+".data:", var)
    else:
        try:
            a = len(var[0])
            print(var_names[0]+":", var[0])
        except:
            print(var_names[0]+":", var)
def ptree(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    print(var_names[0]+".structure:")
    for i in var:
        print('layer1:', type(i),' ')
        try:
            for j in i:
                print('layer2:', type(j),' ')
                try:
                    for k in j:
                        print('layer3:', type(k),' ')
                        try:
                            for k in j:
                                print('layer4:', type(k),' ')
                                break
                        except:
                            return
                        break
                except:
                    return
                break
        except:
            return
        break

def sortDict(dict, By="key", reverse=False):
    if By=="key":
        return sorted(dict.items(), key=lambda x:x[0], reverse=reverse)
    elif By=="value":
        return sorted(dict.items(), key=lambda x:x[1],reverse=reverse)

# 将两个图合并后在同一个图中做GNN更新, 进行测试的方法
class DataSet:
    def __init__(self, root):
        self.root           = root+"/GI"
        self.GI             = self.readGraph(self.root+'/GI/GI.txt', ints=True)
        
        id2vecfile      = self.root+'/GI/entity2vec.bern'
        self.eid2vec    = self.readVec(id2vecfile)
        id2vecfile      = self.root+'/GI/relation2vec.bern'
        self.rid2vec    = self.readVec(id2vecfile)
        self.entity2Cnt = len(self.eid2vec)
        self.relation2Cnt = len(self.rid2vec)
        
        subG1Efile      = root + "/subG1/entity2id.txt"
        self.subG1len   = len(open(subG1Efile).readlines())
        subG2Efile      = root + "/subG2/entity2id.txt"
        self.subG2len   = len(open(subG2Efile).readlines())
        
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
    
    def readVec(self, vf):
        with open(vf, 'r') as f:
            linecnt = 0
            vecs = []
            for line in f:
                vec = [float(i) for i in line.split()]
                vec = np.asarray(vec, dtype=np.float32)
                vecs.append(vec)
            vecs = np.asarray(vecs, dtype=np.float32)
        return vecs
        
    def _Entity2Vec(self, idfile, vecfile):
        entity2id = dict()
        id2vec = dict()
        with open(idfile, 'r') as f:
            for line in f:
                if ' ' in line:
                    entity,id = line.split()
                    id = int(id)
                    entity2id.update({entity:id})
        with open(vecfile, 'r') as f:
            linecnt = 0
            for line in f:
                vec = [float(i) for i in line.split()]
                vec = np.asarray(vec, dtype=np.float32)
                id2vec.update({linecnt:vec})
                linecnt += 1
        return entity2id, id2vec
        
    def id2vec(self):    
        return self.eid2vec, self.rid2vec

    def random(self, positive):
        random1 = np.random.randint(0, self.subG1len-1)
        random2 = np.random.randint(self.subG1len, self.subG1len+self.subG2len-1)
        while([random1,random2] in positive):
            print(random1, random2)
            random1 = np.random.randint(0, self.subG1len-1)
            random2 = np.random.randint(self.subG1len, self.subG1len+self.subG2len-1)
        return random1, random2
    def trainset(self):
        share_nodes_train_file   = self.root+'/subGs/share_nodes_train.txt'
        dataset, positive = [], []
        with open(share_nodes_train_file, 'r') as f:
            for line in f:     
                h,t = [int(i) for i in line.split()]
                # print(h,t)
                positive.append([h,t])
        for i in range(5):
            for h,t in positive:     
                dataset.append([h , t , 1])
                r1, r2 = self.random(positive)
                dataset.append([r1, r2, 0])
        dataset = np.array(dataset)
        np.random.shuffle(dataset)
        return dataset
    
    def testset(self):
        share_nodes_test_file   = self.root+'/subGs/share_nodes_test.txt'
        dataset, positive = [], []
        with open(share_nodes_test_file, 'r') as f:
            for line in f:     
                h,t = [int(i) for i in line.split()]
                positive.append([h,t])
        for h,t in positive:     
            dataset.append([h , t , 1])
            for i in range(2):
                r1, r2 = self.random(positive)
                dataset.append([r1, r2, 0])
        dataset = np.array(dataset)
        np.random.shuffle(dataset)
        return dataset

# 将两个图分别进行训练后, 再用label进行训练的方法.
class DataSet2:
    def __init__(self, root):
        self.root = root
        # entity
        subG1_eidfile    = self.root+'/subG1/entity2id.txt'
        subG1_evecfile   = self.root+'/subG1/entity2vec.bern'
        subG2_eidfile    = self.root+'/subG2/entity2id.txt'
        subG2_evecfile   = self.root+'/subG2/entity2vec.bern'
        self.subG1_entity2id, _ = self._Entity2Vec(subG1_eidfile, subG1_evecfile)
        self.subG2_entity2id, _ = self._Entity2Vec(subG2_eidfile, subG2_evecfile)
        # relation
        subG1_ridfile    = self.root+'/subG1/relation2id.txt'
        subG1_rvecfile   = self.root+'/subG1/relation2vec.bern'
        subG2_ridfile    = self.root+'/subG2/relation2id.txt'
        subG2_rvecfile   = self.root+'/subG2/relation2vec.bern'
        self.subG1_relation2id, _ = self._Entity2Vec(subG1_ridfile, subG1_rvecfile)
        self.subG2_relation2id, _ = self._Entity2Vec(subG2_ridfile, subG2_rvecfile)
        self.subG1_eid2vec = self.readVec(self.root+'/subG1/entity2vec.bern')
        self.subG1_rid2vec = self.readVec(self.root+'/subG1/relation2vec.bern')
        self.subG2_eid2vec = self.readVec(self.root+'/subG2/entity2vec.bern')
        self.subG2_rid2vec = self.readVec(self.root+'/subG2/relation2vec.bern')
        # train test
        train_share_file = self.root+'/train/share_train.txt'
        train_subg1_file = self.root+'/train/subg1_train.txt'
        train_subg2_file = self.root+'/train/subg2_train.txt'
        self.train_share = self._ReadEntity(train_share_file)
        self.train_subg1 = self._ReadEntity(train_subg1_file)
        self.train_subg2 = self._ReadEntity(train_subg2_file)
        test_share_file = self.root+'/test/share_test.txt'
        test_subg1_file = self.root+'/test/subg1_test.txt'
        test_subg2_file = self.root+'/test/subg2_test.txt'
        self.test_share = self._ReadEntity(test_share_file)
        self.test_subg1 = self._ReadEntity(test_subg1_file)
        self.test_subg2 = self._ReadEntity(test_subg2_file)

        self.subG1      = self.readGraph(self.root+'/subG1/train.txt', graph="1")
        self.subG2      = self.readGraph(self.root+'/subG2/train.txt', graph="2")
        self._check()

    def readVec(self, vf):
        with open(vf, 'r') as f:
            linecnt = 0
            vecs = []
            for line in f:
                vec = [float(i) for i in line.split()]
                vec = np.asarray(vec, dtype=np.float32)
                vecs.append(vec)
            vecs = np.asarray(vecs, dtype=np.float32)
        return vecs

    def _Entity2Vec(self, idfile, vecfile):
        entity2id = dict()
        id2vec = dict()
        with open(idfile, 'r') as f:
            for line in f:
                if ' ' in line:
                    entity,id = line.split()
                    id = int(id)
                    entity2id.update({entity:id})
        with open(vecfile, 'r') as f:
            linecnt = 0
            for line in f:
                vec = [float(i) for i in line.split()]
                id2vec.update({linecnt:vec})
                linecnt += 1
        return entity2id, id2vec
    def _ReadEntity(self, file):
        entities = []
        with open(file, 'r') as f:
            all = f.read().split()
            for i in all:                                                   
                entities.append(i)
        return entities
    def _check(self):
        for share in self.test_share:
            try:
                assert(share in self.subG1_entity2id.keys() and share in self.subG2_entity2id.keys())
            except:
                print(share)
    
    def id2vec1(self):
        return self.subG1_eid2vec, self.subG1_rid2vec
    def id2vec2(self):
        return self.subG2_eid2vec, self.subG2_rid2vec
    
    def readGraph(self, sf, graph="1"):
        helper = DataHelper(sf,NP=False)
        samples = helper.GetSamples()
        self.SamplesCnt = len(samples)
        G  = nx.MultiDiGraph()
        if graph=="1":
            for sample in samples:  
                G.add_edge(self.subG1_entity2id[sample[0]],self.subG1_entity2id[sample[2]],
                            attr=self.subG1_relation2id[sample[1]])
        else:
            for sample in samples:  
                G.add_edge(self.subG2_entity2id[sample[0]],self.subG2_entity2id[sample[2]],
                            attr=self.subG2_relation2id[sample[1]])
        return G
    
    def dataset(self):
        trainSet, testSet = [],[]
        # Construct train set
        for k in range(5):
            for i in range(len(self.train_share)):
                if self.train_share[i] in self.subG1_entity2id.keys() \
                    and self.train_share[i] in self.subG2_entity2id.keys():
                    id1 = self.subG1_entity2id[self.train_share[i]]
                    id2 = self.subG2_entity2id[self.train_share[i]]
                    sample = ([id1, id2, 1])
                    trainSet.append(sample)
                    for j in range(2):
                        entity_1 = np.random.choice(self.train_share+self.train_subg1)
                        entity_2 = np.random.choice(self.train_share+self.train_subg2)
                        while(entity_1==entity_2):
                            entity_1 = np.random.choice(self.train_share+self.train_subg1)
                            entity_2 = np.random.choice(self.train_share+self.train_subg2)
                        id1 = self.subG1_entity2id[entity_1]
                        id2 = self.subG2_entity2id[entity_2]
                        sample = ([id1, id2, 0])
                        trainSet.append(sample)
        # Construct test set
        for i in range(len(self.test_share)):
            assert(self.test_share[i] in self.subG1_entity2id.keys() \
                and self.test_share[i] in self.subG2_entity2id.keys())
            id1 = self.subG1_entity2id[self.test_share[i]]
            id2 = self.subG2_entity2id[self.test_share[i]]
            sample = ([id1, id2, 1])
            testSet.append(sample)
            for j in range(2):
                entity_1 = np.random.choice(self.test_share+self.test_subg1)
                entity_2 = np.random.choice(self.test_share+self.test_subg2)
                while(entity_1==entity_2):
                    entity_1 = np.random.choice(self.test_share+self.test_subg1)
                    entity_2 = np.random.choice(self.test_share+self.test_subg2)
                id1 = self.subG1_entity2id[entity_1]
                id2 = self.subG2_entity2id[entity_2]
                sample = ([id1, id2, 0])
                testSet.append(sample)
        # return trainSet
        trainSet = np.array(trainSet)
        testSet = np.array(testSet)
        np.random.shuffle(trainSet)
        np.random.shuffle(testSet)
        return trainSet , testSet

if __name__ == "__main__":    
    root = "/home/dreamer/codes/my_code/pre_based_joint_GI/data/Overlap50.0_TrainRate50.0"
    ds = DataSet2(root)
