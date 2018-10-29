import numpy as np
import sys,traceback,os
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from datahelper import DataHelper
from scipy import stats,optimize
import mylib.texthelper.format as texthelper

def powerNp(x,a):
	b = np.arange(x.shape[0],dtype=np.float)
	for index in range(x.shape[0]):
		b[index]= pow(x[index],a)
	return b

def powerLaw(x,a,b):
	return b*np.power(x,-a)

class GraphMes:
    def __init__(self, logging, graph=None, file=None, start=0, ints=False):
        self.logging = logging
        if not os.path.isdir(logging): os.mkdir(logging)
        
        
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

    def zipf(self, plot=True):
        print('-------------')
        x,y=[],[]
        degree =  nx.degree_histogram(self.G)
        for i in range(len(degree)):
            if degree[i]!=0:
                y.append(degree[i]/float(sum(degree)))
                x.append(i)
        
        xdata = np.array(x)
        ydata = np.array(y)
        fita,fitb = optimize.curve_fit(powerLaw,xdata,ydata)
        print(fita, fitb)
        if plot==False:
            return fita,fitb
        else:
            # x = np.linspace(xdata.min(),xdata.max(),50)
            # y = fita[1]*powerNp(x,-fita[0])
            plt.figure()
            plt.title("Degree distribution curve fitting\n")
            matplotlib.rc('xtick', labelsize=30) 
            matplotlib.rc('ytick', labelsize=30)
            plt.text(max(xdata)*0.4,max(ydata)*0.4,'y='+"{:.3f}".format(fita[1])+'*x^-'+"{:.3f}".format(fita[0]),ha='center')
            plt.plot(xdata,ydata,'.')
            # plt.plot(xdata,ydata,label='data')
            plt.xlabel('k(rank order)')
            plt.ylabel('p(k)')
            plt.savefig(self.logging+'/zipf.png')
            plt.close(0)
            
            plt.figure()
            plt.title("Degree distribution curve fitting (log)\n")
            plt.text(max(xdata)*0.4,max(ydata)*0.4,'y='+"{:.3f}".format(fita[1])+'*x^-'+"{:.3f}".format(fita[0]),ha='center')
            plt.xlabel('k(rank order)')
            plt.ylabel('p(k)')
            plt.loglog(xdata,ydata,'.')
            # plt.loglog(xdata,ydata,'g',label='data')    
            plt.savefig(self.logging+'/zipf_log.png')
            return fita,fitb

    def zipf_coeffi(self, plot=True):
        # print(nx.average_clustering(graphmes.uG))
        degree  = {}
        zipf_coeffi = {}
        for i in self.uG.nodes():
            if self.uG.degree(i) in degree:
                degree[self.uG.degree(i)].append(i) 
            else:
                degree.update({self.uG.degree(i):[i]})
        for i in degree:
            zipf_coeffi.update({i:0})
            for node in degree[i]:
                zipf_coeffi[i] += nx.clustering(self.uG, node)
            zipf_coeffi[i] /= len(degree[i])

        zipf_coeffi = np.array(texthelper.sortDict(zipf_coeffi,By="key"))
        
        if plot == False:
            return zipf_coeffi
        else:
            xdata = zipf_coeffi[:,0]
            ydata = zipf_coeffi[:,1]
            fita,fitb = optimize.curve_fit(powerLaw,xdata,ydata)
            plt.figure()
            plt.title("Degree-Clustering distribution curve fitting\n")
            plt.text(max(xdata)*0.4,max(ydata)*0.4,'y='+"{:.2f}".format(fita[1])+'*x^-'+"{:.2f}".format(fita[0]),ha='center')
            plt.plot(xdata,ydata,'.')
            # plt.plot(xdata,ydata,'.', label='data')
            plt.xlabel('k')
            plt.ylabel('clustering')
            plt.savefig(self.logging+'/zipf_coeffi.png')
            plt.close(0)
            
            plt.figure()
            plt.text(max(xdata)*0.4,max(ydata)*0.4,'y='+"{:.2f}".format(fita[1])+'*x^-'+"{:.2f}".format(fita[0]),ha='center')
            plt.title("Degree-Clustering distribution curve fitting (log)\n")
            plt.loglog(xdata,ydata,'.')
            # plt.loglog(xdata,ydata,'.', label='data')
            plt.xlabel('log(k)')
            plt.ylabel('log(clustering)')
            plt.savefig(self.logging+'/zipf_coeffi_log.png')
            plt.close(0)
            return zipf_coeffi
        
    def record(self, additional=True):
        with open(self.logging+'/info.txt', 'w') as f:
            f.write(" Number of nodes :"+str(len(self.nodes))+'\n')
            f.write(" Number of edges :"+str(len(self.edges))+'\n')
            f.write(" Number of samples :"+str(self.samplesCnt)+'\n')
            if additional:
                uG = nx.Graph(self.G)
                connectedCnt = nx.number_connected_components(uG)
                f.write(" number_connected_components :"+str(connectedCnt)+'\n')
                if connectedCnt == 1:
                    f.write(" Diameter :"+str(nx.diameter(uG))+'\n')
                    f.write(" Radius :"+str(nx.radius(uG))+'\n')
                    f.write(" average_shortest_path_length :"+str(nx.average_shortest_path_length(uG))+'\n')
                f.write(" Density :"+str(nx.density(uG))+'\n')
                f.write(" average_clustering :"+str(nx.average_clustering(uG))+'\n')
                f.write(" node_connectivity :"+str(nx.node_connectivity(self.G))+'\n')
                f.write(" global_efficiency :"+str(nx.global_efficiency(uG))+'\n')
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
    zipf = graphmes.zipf(plot=True)
    zipf_coeffi = graphmes.zipf_coeffi(plot=True)
    