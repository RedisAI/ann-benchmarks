from __future__ import absolute_import
import numpy
import sklearn.preprocessing
from tqdm import tqdm
from ann_benchmarks.algorithms.base import BaseANN


class LSHG(BaseANN):
    def __init__(self, metric, r):
        self._r = r
        self._metric = metric
        self._configs =  None
        self._X = None
        self._dim = None
        self._n = None
        self._signatures = None
        self._name = "LSHG"
        self._diags = None
        
    def __str__(self):
        return(f"r={self._r}")

    def fwht(self,a) -> None:
        h = 1
        while h < len(a):
            # perform FWHT
            for i in range(0, len(a), h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            # normalize and increment
            a = a/2
            h *= 2
    
    def diagonals(self,r):
        res=[]
        for i in range(r):
            d1=[]
            d2=[]
            d3=[]
            for j in range(self._dim):
                x1=numpy.random.rand()
                x2=numpy.random.rand()
                x3=numpy.random.rand()
                if x1>.5:
                    d1.append(1)
                else:
                    d1.append(-1)
                if x2>.5:
                    d2.append(1)
                else:
                    d2.append(-1)
                if x3>.5:
                    d3.append(1)
                else:
                    d3.append(-1)
            res.append([d1,d2,d3])
        return(res)

    def imax(self,L):
        m=abs(L[0])
        if L[0]>=0:
            L[0]=1
        else:
            L[0]=-1
        imax=0
        for i in range(1,len(L)):
            if abs(L[i])>m:
                m=abs(L[i])
                L[imax]=0
                imax=i
                if L[i]>=0:
                    L[i]=1
                else:
                    L[i]=-1    
            else:
                L[i]=0
    
    def signvect(self, x):
        res=[]
        for i in range(self._r):
            [d1,d2,d3]=self._diags[i]
            u=numpy.copy(x)
            u=numpy.multiply(d1,u)
            self.fwht(u)
            u=numpy.multiply(d2,u)
            self.fwht(u)
            u=numpy.multiply(d3,u)
            self.fwht(u)
            self.imax(u)
            res.append(u)
        return(res)

    def signature(self, data, r):
        res=[]
        for i in range(r):
            [d1,d2,d3]=self._diags[i]
            for x in tqdm(data):
                u=numpy.copy(x)
                u=numpy.multiply(d1,u)
                self.fwht(u)
                u=numpy.multiply(d2,u)
                self.fwht(u)
                u=numpy.multiply(d3,u)
                self.fwht(u)
                self.imax(u)
                res.append(u)
        return(res)

    def fit(self, X):
        self._n, self._dim=numpy.shape(X)
        self._diags = self.diagonals(self._r)
        print("fit")
        self._X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        X=self._X
        self._signatures = self.signature(self._X, self._r) 
        
        self._configs=numpy.zeros((self._r,2*self._dim),dtype=list)
        for i in range (self._r):
            for j in range(2*self._dim):
                self._configs[i][j]=[]
        for i in (range(self._n)):
            for j in range(self._r):
                s=self._signatures[i+j*self._n]
                for p in range(self._dim):
                    if s[p]==1:
                        k=p
                    if s[p]==-1:
                        k= self._dim+p
                (self._configs[j][k]).append(i)
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
        v /= numpy.linalg.norm(v)
        vsign = self.signvect(v)
        resid=[]
        for j in range(self._r):
              resid = list(set(resid)| set(self._configs[j][k]))
        return(self.BF(v, k, resid))

