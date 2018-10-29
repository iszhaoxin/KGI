import numpy as np
import sys, os
import chainer
import inspect, json
import chainer.functions as F
import chainer.links as L
from multiprocessing import Pool
from chainer import reporter, Variable, Chain, Sequential, link, Parameter
from chainer import datasets, iterators, optimizers, training
from chainer.training import extensions
from MaskLinear import MaskLinear
from chainer.initializers.uniform import Uniform


class DataSet:
    def __init__(self, root):
        self.root = root
        subG1_idfile = self.root+'/subG1/entity2id.txt'
        subG1_vecfile = self.root+'/subG1/entity2vec.bern'
        subG2_idfile = self.root+'/subG2/entity2id.txt'
        subG2_vecfile = self.root+'/subG2/entity2vec.bern'
        self.subG1_entity2id, self.subG1_id2vec = self._Entity2Vec(subG1_idfile, subG1_vecfile)
        self.subG2_entity2id, self.subG2_id2vec = self._Entity2Vec(subG2_idfile, subG2_vecfile)
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

        self._check()
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
    def __call__(self):
        # print(self.train_share+self.train_subg2)
        # print('------------')
        # print(self.subG2_entity2id.keys())
        trainSet, testSet = [],[]
        # Construct train set
        for i in range(len(self.train_share)):
            if self.train_share[i] in self.subG1_entity2id.keys() \
                and self.train_share[i] in self.subG2_entity2id.keys():
                id1 = self.subG1_entity2id[self.train_share[i]]
                vec1 = self.subG1_id2vec[id1]
                id2 = self.subG2_entity2id[self.train_share[i]]
                vec2 = self.subG2_id2vec[id2]
                sample = (np.array(vec1+vec2).astype(np.float32),1)
                trainSet.append(sample)
                for j in range(1):
                    entity_1 = np.random.choice(self.train_share+self.train_subg1)
                    entity_2 = np.random.choice(self.train_share+self.train_subg2)
                    while(entity_1==entity_2):
                        entity_1 = np.random.choice(self.train_share+self.train_subg1)
                        entity_2 = np.random.choice(self.train_share+self.train_subg2)
                    id1 = self.subG1_entity2id[entity_1]
                    vec1 = self.subG1_id2vec[id1]
                    id2 = self.subG2_entity2id[entity_2]
                    vec2 = self.subG2_id2vec[id2]
                    sample = (np.array(vec1+vec2).astype(np.float32),0)
                    trainSet.append(sample)
        # Construct test set
        for i in range(len(self.test_share)):
            assert(self.test_share[i] in self.subG1_entity2id.keys() \
                and self.test_share[i] in self.subG2_entity2id.keys())
            id1 = self.subG1_entity2id[self.test_share[i]]
            vec1 = self.subG1_id2vec[id1]
            id2 = self.subG2_entity2id[self.test_share[i]]
            vec2 = self.subG2_id2vec[id2]
            sample = (np.array(vec1+vec2).astype(np.float32),1)
            testSet.append(sample)
            for j in range(10):
                entity_1 = np.random.choice(self.test_share+self.test_subg1)
                entity_2 = np.random.choice(self.test_share+self.test_subg2)
                while(entity_1==entity_2):
                    entity_1 = np.random.choice(self.test_share+self.test_subg1)
                    entity_2 = np.random.choice(self.test_share+self.test_subg2)
                id1 = self.subG1_entity2id[entity_1]
                vec1 = self.subG1_id2vec[id1]
                id2 = self.subG2_entity2id[entity_2]
                vec2 = self.subG2_id2vec[id2]
                sample = (np.array(vec1+vec2).astype(np.float32),0)
                testSet.append(sample)
        # return trainSet
        np.random.shuffle(trainSet)
        # np.random.shuffle(testSet)
        return trainSet , testSet

class MaskConcatel(chainer.Chain):
    def __init__(self, hidden_unit):
        super(MaskConcatel, self).__init__()
        with self.init_scope():
            self.l1 = MaskLinear(200, 2*hidden_unit, direction='l')
            self.l2 = MaskLinear(2*hidden_unit, 2, direction='l')

    def __call__(self, x):
        a = self.l1(x)
        x1 = F.dropout(x, ratio=0.5)
        h1 = F.relu(self.l1(x))
        x2 = F.dropout(h1, ratio=0.5)
        return self.l2(x2)

class MaskConcater(chainer.Chain):
    def __init__(self, hidden_unit):
        super(MaskConcater, self).__init__()
        with self.init_scope():
            self.l1 = MaskLinear(200, 2*hidden_unit, direction='r')
            self.l2 = MaskLinear(2*hidden_unit, 2, direction='r')

    def __call__(self, x):
        a = self.l1(x)
        x1 = F.dropout(x, ratio=0.5)
        h1 = F.relu(self.l1(x))
        x2 = F.dropout(h1, ratio=0.5)
        return self.l2(x2)

class ShallowUnConcate(Chain):
    def __init__(self,hidden_unit):
        super(ShallowUnConcate, self).__init__()
        with self.init_scope():
            if hidden_unit%2!=0:
                raise EOFError
            self.layers = []
            self.ll1 = L.Linear(100, int(hidden_unit))
            self.lr1 = L.Linear(100, int(hidden_unit))
            self.l2 = L.Linear(2*hidden_unit, 2)

    def __call__(self, x):
        x1 = x[:,:100]
        x1_dropout = F.dropout(x1, ratio=0.5)
        x2 = x[:,100:]
        x2_dropout = F.dropout(x2, ratio=0.5)
        hl1 = F.relu(self.ll1(x1_dropout))
        hr1 = F.relu(self.lr1(x2_dropout))
        h2 = F.dropout(F.concat((hl1, hr1),axis=1))
        # print(self.l2.W[0][0])
        return self.l2(h2)

class ShallowConcate(chainer.Chain):
    def __init__(self, hidden_unit):
        super(ShallowConcate, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None , hidden_unit) 
            self.l2 = L.Linear(None , 2)
            
    def __call__(self, x):
        x1 = F.dropout(x, ratio=0.5)
        h1 = F.relu(self.l1(x))
        x2 = F.dropout(h1, ratio=0.5)
        return self.l2(x2)

class ShallowConcateS(chainer.Chain):
    def __init__(self, hidden_units):
        print(hidden_units)
        super(ShallowConcateS, self).__init__()
        with self.init_scope():
            self.layers = Sequential()
            for layer_units in hidden_units:
                self.layers.append(L.Linear(None, layer_units))
                self.layers.append(F.relu)
                self.layers.append(L.BatchNormalization(layer_units))
                self.layers.append(F.dropout)
            self.last = L.Linear(None, 2)
        print(self.layers)

    def __call__(self, x):
        y = self.layers(x)
        return self.last(y)

class npUnconcat(chainer.Chain):
    def __init__(self, hidden_units):
        super(npUnconcat, self).__init__()
        with self.init_scope():
            initializer     = Normal()

            self.encoderL = L.Linear(None , hidden_units[0])
            self.encoderR = L.Linear(None , hidden_units[0])
            self.z        = Parameter(initializer)
            self.z.initialize(hidden_units[1])
            self.decoderL = L.Linear(None , hidden_units[2])
            self.decoderR = L.Linear(None , hidden_units[2])
        

class Classifier(link.Chain):
    compute_accuracy = True
    def __init__(self, predictor,
                 lossfun=F.softmax_cross_entropy,
                 accfun=F.accuracy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(Classifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

    def forward(self, *args, **kwargs):
        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy   = F.accuracy(self.y, t)
            self.precision, self.recall,self.f1_score = F.classification_summary(self.y, t)[:3]
            reporter.report({'recall': self.recall[1]}, self)
            reporter.report({'accuracy': self.accuracy}, self)
            reporter.report({'f1_score': self.f1_score[1]}, self)
            reporter.report({'precision': self.precision[1]}, self)
        return self.loss
