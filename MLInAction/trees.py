from math import log
import operator

# Function to calculate the Shannon entropy of dataset
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    ## Create dictionary of all possible classes
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # Logarithm base 2
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# Dataset splitting on a given feature
def splitDataSet(dataSet, axis, value):
    retDataSet = [] # Create separate list
    ## Cut out the feature split on
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# Choosing the best feature to split on
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # Create unique list of class labels
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        ## Calculate entropy for each split
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        ## Find the best information gain
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# Tree-building code
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    ## Stop when all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    ## When no more features, return majority
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    ## Get list of unique values
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # this makes a copy of labels and places it in a new list subLabels
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def createTree_debug(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    print 'classList', classList
    print 'dataSet[0]:', dataSet[0]
    ## Stop condition1: when all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    ## Stop condition2: when no more features, return majority
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    ## Get list of unique values
    print 'labels', labels
    del(labels[bestFeat])
    print 'labels', labels
    featValues = [example[bestFeat] for example in dataSet]
    print 'featValues', featValues
    uniqueVals = set(featValues)
    print 'uniqueVals', uniqueVals
    for value in uniqueVals:
        subLabels = labels[:]
        print 'labels', labels
        print 'subLabels', subLabels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


myDat, labels = createDataSet()
print 'myDat', myDat
print 'labels', labels
print calcShannonEnt(myDat)
print splitDataSet(myDat, 0, 0)
print chooseBestFeatureToSplit(myDat)
myTree = createTree_debug(myDat, labels)
print myTree