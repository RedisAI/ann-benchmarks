from __future__ import absolute_import
import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN


class LSH(BaseANN):
    def __init__(self, metric, b, r):
        self._b = b
        self._r = r
        self._metric = metric
        self._configs =  None
        self._V= None
        self._X = None
        self._dim = None
        self._n = None
        self._signatures = None
        self.name = "LSH"
        
    def __str__(self):
        return(f"b={self._b}, r={self._r}")

    def binar(self, vector):
        res=0
        for k in range(len(vector)):
            if vector[k]>=0:
                res+=2**k 
        return(res)

    def fit(self, X):
        print("fit")
        self._n, self._dim=numpy.shape(X)
        self._X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        X=self._X
        self._V=numpy.random.randn(self._b*self._r,self._dim)
        self._signatures =  numpy.matmul(X,numpy.matrix.transpose(self._V))
        self._configs=numpy.zeros((self._r,2**self._b),dtype=list)
        for i in range (self._r):
            for j in range(2**self._b):
                self._configs[i][j]=[]
        for i in (range(self._n)):
            for j in range(self._r):
                self._configs[j][self.binar(self._signatures[i][j*self._b:(j+1)*self._b])].append(i)
        
        print("fit")
    
    def cosdis(self, x,y):
        return(1-numpy.dot(x,y))


    def BF(self, v, k, L):
        res=[]
        for i in range(len(L)):
            res.append([self.cosdis(v, self._X[L[i]]),L[i]])
        res.sort(key=lambda element:element[0])
        return([i[1] for i in res[:k]])

    def query(self, v, k):
        configs = self._configs
        v /= numpy.linalg.norm(v)
        vsign = numpy.matmul(v,numpy.matrix.transpose(self._V))
        resid=[]
        for j in range(self._r):
            resid = list(set(resid)| set(configs[j][self.binar(vsign[j*self._b:(j+1)*self._b])]))
        return(self.BF(v, k, resid))

