# Copied code from: Kevin Duh
# https://github.com/kevinduh/pareto

#!/usr/bin/env python
class Datapoint:
    """Defines a point in K-dimensional space"""
    def __init__(self,id):
        self.id = id # datapoint id (0,..N-1)
        self.vec = [] # the K-dim vector
        self.paretoStatus = -1 # -1=dont know, 1=pareto, 0=not pareto
        self.dominatedCount = 0 # number of datapoints that dominate this point
        self.dominatingSet = [] # set of vectors this one is dominating

    def addNumber(self,num):
        """Adds a number to one dimension of this datapoint"""
        self.vec.append(num)

    def addToDominatingSet(self,id2):
        """Add id of of dominating point"""
        self.dominatingSet.append(id2)

    def dominates(self,other):
        """Returns true if self[k]>=other[k] for all k and self[k]>other[k] for at least one k"""
        assert isinstance(other,Datapoint)
        gte=0 # count of self[k]>=other[k]
        gt=0 # count of self[k]>other[k]
        for k in range(len(self.vec)):
            if self.vec[k] >= other.vec[k]:
                gte+=1
                if self.vec[k] > other.vec[k]:
                    gt+=1

        return (gte==len(self.vec) and (gt>0))

    def __repr__(self):
        return self.vec.__repr__()+": "+str(self.paretoStatus)

def nondominated_sort(dataset):
    """Nondominated Sorting, generates ranking w/ higher number = better pareto front"""
    numPareto = 0

    # pairwise comparisons
    for n in range(len(dataset)):
        for m in range(len(dataset)):
            if dataset[m].dominates(dataset[n]):
                dataset[n].dominatedCount+=1
                dataset[m].addToDominatingSet(n)

    # find first pareto front
    front = []
    front2 = []
    tmpLevel = -10 # temporary value for Pareto level, will re-adjust later
    for n in range(len(dataset)):
        if dataset[n].dominatedCount == 0:
            dataset[n].paretoStatus = tmpLevel
            front.append(n)
            numPareto+=1

    # iteratively peel off pareto fronts
    while len(front) != 0:
        tmpLevel-=1
        for f in front:
            for s in dataset[f].dominatingSet:
                dataset[s].dominatedCount -= 1
                if dataset[s].dominatedCount == 0:
                    front2.append(s)
                    dataset[s].paretoStatus = tmpLevel
        front = front2
        front2 = []

    # re-adjust pareto level
    for n in range(len(dataset)):
        oldLevel = dataset[n].paretoStatus
        if oldLevel != -1:
            dataset[n].paretoStatus = oldLevel-tmpLevel-1

    return numPareto

def create_dataset(raw_vectors):
    """Given a list of vectors, create list of datapoints"""
    dataset = []
    for k in range(len(raw_vectors)):
        for n,v in enumerate(raw_vectors[k]):
            if k == 0:
                dataset.append(Datapoint(n))
            dataset[n].addNumber(v)
    return dataset


def pareto(y, opt=[-1,-1]):
    # y: np.ndarray(n_y, d)
    # larger is better: opt = 1
    # smaller is better: opt = -1
    new_y = []
    for i in range(len(opt)):
        o = opt[i]
        new_y.append(o*y[i])

    dataset = create_dataset(new_y)
    nondominated_sort(dataset)

    ranking = [dp.paretoStatus for dp in dataset]

    return ranking
