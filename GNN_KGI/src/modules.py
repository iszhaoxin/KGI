import numpy as np
import json, inspect
from functools import partial
from itertools import repeat
import sys, os, time, datetime, shutil
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as C
from graph import *
from datahelper import *
from dataset import DataSet, DataSet2
from multiprocessing import Pool
from scipy.spatial.distance import cosine
from mylib.texthelper.format import pshape, pdata, ptree
from chainer.training import extensions
from chainer.initializers.normal import Normal
from chainer import reporter, Variable, Chain, Parameter, link
from chainer import datasets, iterators, optimizers, training, serializers

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def print_withname(variable):
    variable_name = retrieve_name(variable)
    print(variable_name+'\'s type: '+type(variable))

def _check_index(pre, suc):
    assert(len(pre)==len(suc))
    for i in range(len(pre)):
        assert(len(pre[i])>0 or len(suc[i])>0)

def copy_file(src, dst):
    shutil.copy(src, dst)

class NVec(link.Link):
    def __init__(self, nodeVecs, vecDims, dropout_ratio):
        super(NVec, self).__init__()
        with self.init_scope():
            self.vecDims    = vecDims
            self.nodeVecs   = Parameter(nodeVecs)
            self.dropout_ratio = dropout_ratio
    def forward(self, indexs):
        mask = np.random.rand(len(indexs)) >= self.dropout_ratio
        mask = mask*1 
        vecs = F.embed_id(indexs, self.nodeVecs).reshape(-1, self.vecDims)
        # vecs = F.einsum('ij,i->ij', vecs, mask)
        vecs = vecs.T*mask
        vecs = vecs.T
        return vecs

class RVec(chainer.link.Chain):
    def __init__(self, edgeVecs, vecDims, dropout_ratio):
        super(RVec, self).__init__()
        with self.init_scope():
            self.vecDims    = vecDims
            self.edge2vec   = Parameter(edgeVecs)
            self.dropout_ratio = dropout_ratio
    def forward(self, indexs):
        # print("self.edge2vec:",self.edge2vec[0][0])
        mask = np.random.rand(len(indexs)) >= self.dropout_ratio
        mask = mask*1 
        vecs = F.embed_id(indexs, self.edge2vec).reshape(-1, self.vecDims)
        vecs = vecs.T*mask
        vecs = vecs.T
        return vecs

class Trans(chainer.link.Chain):
    def __init__(self, hidden_unit, output_unit, dropout, linear=False):
        super(Trans, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            self.l1 = L.Linear(200, hidden_unit)
            self.l2 = L.Linear(hidden_unit, output_unit)
            self.linear = linear
    def __call__(self, x):
        if self.linear == True:
            h1 = self.l1(x)
            return self.l2(h1)
        else:
            x1 = F.dropout(x, ratio=0.5)
            h1 = F.relu(self.l1(x))
            x2 = F.dropout(h1, ratio=0.5)
        return self.l2(h1)

# Through a nerual network
class Merger(chainer.link.Chain):
    def __init__(self, dim, dropout_rate, activate, isR, isBN):
        super(Merger, self).__init__()
        with self.init_scope():
            self.is_residual    = isR
            self.is_batchnorm   = isBN
            self.activate       = activate
            self.dim            = dim
            self.dropout_rate   = dropout_rate
            self.x2z	        = L.Linear(dim,dim)
            self.bn	            = L.BatchNormalization(dim)
            self.pool           = Pool(8)
            # self.pool           = Pool(settings['CPUs'])
    def forward(self, x, index_array):
        x = x.reshape(-1, self.dim)
        if self.dropout_rate!=0:
            x = F.dropout(x,ratio=self.dropout_rate)
        z = self.x2z(x)
        if self.activate=='tanh':
            z = F.tanh(z)
        if self.activate=='relu':
            z = F.relu(z)
        if self.is_residual:
            z = z+x
        split_array = F.split_axis(z, index_array, axis=0)[:-1]
        a = []
        for i in split_array:
            if len(i)>0:
                a.append(F.average(i,axis=0))
            else:
                a.append(Variable(np.zeros(self.dim, dtype=np.float32)))
        p = F.stack(a)
        return p

# only average
class Merger2(chainer.link.Chain):
    def __init__(self):
        super(Merger2, self).__init__()
        with self.init_scope():
            self.pool           = Pool(8)
    def forward(self, x, index_array):
        split_array = F.split_axis(x, index_array, axis=0)[:-1]
        a = []
        for i in split_array:
            if len(i)>0:
                a.append(F.average(i,axis=0))
            else:
                a.append(Variable(np.zeros(self.dim, dtype=np.float32)))
        p = F.stack(a)
        return p

class GNN(chainer.link.Chain):
    def __init__(self, ds, settings):
        super(GNN, self).__init__()
        with self.init_scope():
            # Settings
            self.settings = settings
            self.pooling    = F.average
            self.CPUs       = settings["CPUs"]
            self.vecDims    = settings['vecDims']
            self.dropout    = settings["dropout"]
            self.batch_size = settings["batch_size"]
            self.self_ratio = settings["self_ratio"]
            self.PTrans     = Trans(settings["hiddenUnits_P"], 100, self.dropout, linear=settings['linear_PS'])
            self.STrans     = Trans(settings["hiddenUnits_S"], 100, self.dropout, linear=settings['linear_PS'])
            self.CTrans     = Trans(settings["hiddenUnits_C"], 2,   self.dropout, linear=settings['linear_C'])
            # graph mes
            self.subG1_mes = GraphMes(ds.subG1)
            self.subG2_mes = GraphMes(ds.subG2)
            self.subG1_mes.tensor()
            self.subG2_mes.tensor()
            # dataset setting
            # self.node2vec, self.edgevec = self.ds.id2vec()
            self.node2vec1, self.edgevec1 = ds.id2vec1()
            self.node2vec2, self.edgevec2 = ds.id2vec2()
            self.nvec1      = NVec(self.node2vec1, self.vecDims, dropout_ratio=settings['dropout_vec'])
            self.rvec1      = RVec(self.edgevec1,  self.vecDims, dropout_ratio=settings['dropout_vec'])
            self.nvec2      = NVec(self.node2vec2, self.vecDims, dropout_ratio=settings['dropout_vec'])
            self.rvec2      = RVec(self.edgevec2,  self.vecDims, dropout_ratio=settings['dropout_vec'])
            # model modules
            self.merger     = Merger(self.vecDims, self.dropout, 'relu', isR=True, isBN=True)
            # self.merger     = Merger2()
            
    def forward(self, x):
        hdx     = x[:,0]
        tdx     = x[:,1]
        hvec    = self.propagation(hdx, "1")
        tvec    = self.propagation(tdx, "2")
        concat  = F.concat((hvec, tvec), axis=1)
        return self.CTrans(concat)

    def propagation(self, idxs, graph):
        if graph == "1":
            mes = self.subG1_mes.tensor2(idxs)
            ivecs = self.nvec1(idxs)
        elif graph == "2":
            mes = self.subG2_mes.tensor2(idxs)
            ivecs = self.nvec2(idxs)
        
        pre_mes = mes[0][0]
        suc_mes = mes[1][0]
        pre_pre_mes, pre_suc_mes = mes[0][1]
        suc_pre_mes, suc_suc_mes = mes[1][1]

        pre_dot = self.dot(pre_pre_mes, pre_suc_mes, graph)
        suc_dot = self.dot(suc_pre_mes, suc_suc_mes, graph)
        if graph == "1":
            pre_r_vecs  = self.rvec1(pre_mes.links)
            suc_r_vecs  = self.rvec1(suc_mes.links)
        elif graph == "2":
            pre_r_vecs  = self.rvec2(pre_mes.links)
            suc_r_vecs  = self.rvec2(suc_mes.links)
        pre_concat  = F.concat((pre_dot,pre_r_vecs),axis=1)
        pre_dot2    = self.PTrans(pre_concat)
        pre_dot2    = self.self_ratio*pre_dot2 + (1-self.self_ratio)*pre_r_vecs
        suc_concat  = F.concat((suc_dot,suc_r_vecs),axis=1)
        suc_dot2    = self.STrans(suc_concat)
        suc_dot2    = self.self_ratio*suc_dot2 + (1-self.self_ratio)*suc_r_vecs
        
        if pre_dot is not None and suc_dot is not None:
            pre_dot_group = F.split_axis(pre_dot2, pre_mes.indexs, axis=0)[:-1]
            suc_dot_group = F.split_axis(suc_dot2, suc_mes.indexs, axis=0)[:-1]
        elif pre_dot is not None:
            pre_dot_group = F.split_axis(pre_dot2, pre_mes.indexs, axis=0)[:-1]
            suc_dot_group = None
        elif suc_dot is not None:
            suc_dot_group = F.split_axis(suc_dot2, suc_mes.indexs, axis=0)[:-1]
            pre_dot_group = None
        dot = self.caculate(pre_dot_group, suc_dot_group)
        dot = self.self_ratio*ivecs + (1-self.self_ratio)*dot

        return dot
        
    def dot(self, pre_mes, suc_mes, graph):
        if suc_mes.nodes == [] and pre_mes.nodes == []:
            return None
        if pre_mes.nodes != []:
            if graph == "1":
                pre_n_vecs      = self.nvec1(pre_mes.nodes)
                pre_r_vecs      = self.rvec1(pre_mes.links)
            elif graph == "2":
                pre_n_vecs      = self.nvec2(pre_mes.nodes)
                pre_r_vecs      = self.rvec2(pre_mes.links)
            pre_concat      = F.concat((pre_n_vecs,pre_r_vecs),axis=1)
            pre_dot         = self.PTrans(pre_concat)
            pre_dot_group   = F.split_axis(pre_dot, pre_mes.indexs, axis=0)[:-1]
        else:
            pre_dot_group = None
        if suc_mes.nodes != []:
            if graph == "1":
                suc_n_vecs      = self.nvec1(suc_mes.nodes)
                suc_r_vecs      = self.rvec1(suc_mes.links)
            elif graph == "2":
                suc_n_vecs      = self.nvec2(suc_mes.nodes)
                suc_r_vecs      = self.rvec2(suc_mes.links)
            suc_concat      = F.concat((suc_n_vecs,suc_r_vecs),axis=1)
            suc_dot         = self.STrans(suc_concat)
            suc_dot_group   = F.split_axis(suc_dot, suc_mes.indexs, axis=0)[:-1]
        else:
            suc_dot_group = None

        dot = self.caculate(pre_dot_group, suc_dot_group)
        return dot

    def caculate(self, pre, suc):
        arrays = []
        arrays_index = []
        index = 0
        if pre!=None and suc!=None:
            for i in range(len(pre)):
                array = F.concat((pre[i],suc[i]), axis=0)                
                arrays.append(array)
                index += array.shape[0]
                arrays_index.append(index)
        elif pre==None and suc!=None:
            for i in range(len(suc)):
                arrays.append(suc[i])
                index += suc[i].shape[0]
                arrays_index.append(index)
        elif suc==None and pre!=None:
            for i in range(len(pre)):
                arrays.append(pre[i])
                index += pre[i].shape[0]
                arrays_index.append(index)
        else:
            return None
        arrays = F.concat(arrays, axis=0)
        arrays_index = np.array(arrays_index)
        dot_array = self._caculate(arrays, arrays_index)
        return dot_array

    def _caculate(self, arrays, arrays_index):
        if len(arrays) > 0:
            result = self.merger(arrays, arrays_index)
            
        else:
            return Variable(np.zeros(self.vecDims, dtype=np.float32))
        return result

class Classifier(link.Chain):
    def __init__(self, predictor,
                 lossfun=F.softmax_cross_entropy,
                 accfun=F.accuracy, compute_accuracy=1):
        super(Classifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.compute_accuracy = compute_accuracy
        with self.init_scope():
            self.predictor = predictor

    def forward(self, x):
        xp  = np.array(x)
        x   = xp[:,:2]
        t   = xp[:,2]
        self.y = self.predictor(x)
        self.loss = self.lossfun(self.y, t).reshape(1)
        reporter.report({'loss': self.loss[0]}, self)
        if self.compute_accuracy:
            self.accuracy   = F.accuracy(self.y, t)
            # self.precision, self.recall,self.f1_score = F.classification_summary(self.y, t)[:3]
            # reporter.report({'recall': self.recall[1]}, self)
            reporter.report({'accuracy': self.accuracy}, self)
            # reporter.report({'f1_score': self.f1_score[1]}, self)
            # reporter.report({'precision': self.precision[1]}, self)
        return self.loss.reshape(1)

def train(root):
    settings = Settings()
    print("batch_size:"         , settings['batch_size'])
    print("epoch: "             , settings['epoch'])
    print("alpha: "             , settings['alpha'])
    print("beta1: "             , settings['beta1'])
    print("beta2: "             , settings['beta2'])
    print("eps: "               , settings['eps'])
    print("vecDims: "           , settings['vecDims'])
    print("is_parallel: "       , settings['is_parallel'])
    print("overlap_rate: "      , settings['overlap_rate'])
    print("train_rate: "        , settings['train_rate'])
    print("dropout: "           , settings['dropout'])
    print("dropout_vec: "       , settings['dropout_vec'])
    print("hiddenUnits_P: "     , settings['hiddenUnits_P'])
    print("hiddenUnits_S: "     , settings['hiddenUnits_S'])
    print("hiddenUnits_C: "     , settings['hiddenUnits_C'])
    
    ds = DataSet2(root)
    trainSet, testSet    = ds.dataset()
    train_iterator  = chainer.iterators.SerialIterator(trainSet, settings['batch_size'], repeat=True)
    test_iterator  = chainer.iterators.SerialIterator(testSet, settings['batch_size'], repeat=False)

    model = Classifier(GNN(ds, settings))
    
    # for x in train_iterator:
    #     y = model(x)
        # g = C.build_computational_graph(y, show_name=False)
    #     with open('./com_graph1.dot', 'w') as o:
    #         o.write(g.dump())
    #     break
    
    optimizer = optimizers.Adam(alpha=settings['alpha'],beta1=settings['beta1'],beta2=settings['beta1'],eps=settings['eps'])
    optimizer.setup(model)
    updater = training.updater.StandardUpdater(train_iterator,optimizer)
    
    dt_now = datetime.datetime.now()
    time_stample = '../result/'+root.split('/')[-1]+str(dt_now.day)+'_'+str(dt_now.hour)+'_'+str(dt_now.minute)+'_'+str(dt_now.second)+'/'
    os.mkdir(time_stample)
    copy_file('../model/settings.json', time_stample+'settings.json') 
    trainer = training.Trainer(updater,(settings['iteration'], 'iteration'), out=time_stample)

    trainer.extend(extensions.Evaluator(test_iterator, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    # trainer.extend(extensions.PrintReport(['validation/main/loss', 'validation/main/accuracy',
    # 'validation/main/precision','validation/main/f1_score','validation/main/recall']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],'iteration',
                trigger=(1, 'iteration'), file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'iteration',
                trigger=(1, 'iteration'), file_name='accuracy.png'))
    trainer.extend(extensions.PlotReport(['main/precision', 'validation/main/precision'],'iteration', 
                trigger=(1, 'iteration'),file_name='precision.png'))
    trainer.extend(extensions.PlotReport(['main/f1_score', 'validation/main/f1_score'],'iteration', 
                trigger=(1, 'iteration'),file_name='f1_score.png'))
    trainer.extend(extensions.PlotReport(['main/recall', 'validation/main/recall'],'iteration', 
                trigger=(1, 'iteration'),file_name='recall.png'))

    # trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == "__main__":
    # p = Pool(5)
    # files = ["../data/WNM_O50T50", "../data/WNB_O50T50", "../data/FBM_O50T50", "../data/FBB_O50T50",
    # "../data/FBS_O50T50"]
    # p.map(train, files)
    root = "../data/WNM_Overlap50.0_TrainRate50.0"
    # files = ["../data/WNM_O50T50"]
    train(root)