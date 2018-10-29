import numpy as np
import sys, os
import chainer
import inspect, json
import chainer.functions as F
import chainer.links as L
from multiprocessing import Pool
from chainer import reporter, Variable, Chain, Sequential
from chainer import datasets, iterators, optimizers, training
from chainer.training import extensions
from modules import DataSet, MaskConcatel, MaskConcater, ShallowUnConcate, ShallowConcate, ShallowConcateS, Classifier



def Settings():
    with open('./settings.json') as f:
        settings = json.load(f)
    return settings

def train(model_VALUE, hidden_unit, root):
    # print(root)
    trainSet, testSet = DataSet(root)()
    train_iterator = chainer.iterators.SerialIterator(trainSet, settings['batch_size'])
    test_iterator  = chainer.iterators.SerialIterator(testSet, settings['batch_size'], repeat=False)

    if model_VALUE == "MaskConcatel":
        model = Classifier(MaskConcatel(hidden_unit), lossfun=F.softmax_cross_entropy)
    elif model_VALUE == "MaskConcater":
        model = Classifier(MaskConcater(hidden_unit), lossfun=F.softmax_cross_entropy)
    elif model_VALUE == "ShallowUnConcate":
        model = Classifier(ShallowUnConcate(hidden_unit), lossfun=F.softmax_cross_entropy)
    elif model_VALUE == "ShallowConcate":
        model = Classifier(ShallowConcate(hidden_unit), lossfun=F.softmax_cross_entropy)
    elif model_VALUE == "ShallowConcateS":
        model = Classifier(ShallowConcateS(hidden_unit), lossfun=F.softmax_cross_entropy)
        
    optimizer = optimizers.SGD(settings['learning_rate'])
    optimizer.setup(model)
    updater = training.updater.StandardUpdater(train_iterator,optimizer)
#     trainer = training.Trainer(updater,(settings['epoch'], 'epoch'), out=root+'/result/'+str(model_VALUE)+'/'+str(hidden_unit))
    trainer = training.Trainer(updater,(settings['epoch'], 'epoch'), out=root+'/result/'+str(model_VALUE)+'/'+str(len(hidden_unit)))
    trainer.extend(extensions.Evaluator(test_iterator, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/recall', 'validation/main/recall'],'epoch', file_name='recall.png'))
    trainer.extend(extensions.PlotReport(['main/f1_score', 'validation/main/f1_score'],'epoch', file_name='f1_score.png'))
    trainer.extend(extensions.PlotReport(['main/precision', 'validation/main/precision'],'epoch', file_name='precision.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

if __name__ == "__main__":
    settings = Settings()
    print(settings)
    print("batch_size: ", settings['batch_size'])
    print("test_size: " , settings['test_size'])
    print("epoch: "     , settings['epoch'])
    print("learning_rate: ", settings['learning_rate'])

    # def main(set):
    #     root = '/home/dreamer/codes/my_code/KGI/sampling/results/fb15k/OverRate50_TrainRate70/'
    #     for directory in os.listdir(root):
    #         if os.path.isdir(root+directory):
    #             train(set[0], set[1],root+directory)
    
    root = "/home/dreamer/codes/my_code/KGI/sampling/results/fb15k/50_70_4/uniform"
    train("MaskConcatel", 50 , root)
    # p = Pool(3)
    # layerss = []
    # layerss.append(["MaskConcatel", 50])
    # layerss.append(["ShallowUnConcate", 50])
    # layerss.append(["ShallowConcate", 50])
    # p.map(main, layerss)

    # p = Pool(4)
    # sets = [["ShallowConcate",50]]
    # p.map(main, sets)
    