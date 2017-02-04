import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=False)
    return sortedClassCount[0][0]


def classify0_debug(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=False)

    print 'sqDiffMat:', sqDiffMat
    print 'sqDistances axis = 0:', sqDiffMat.sum(axis=0)
    print 'sqDistances axis = 1:', sqDiffMat.sum(axis=1)
    print 'sqDistances axis = none:', sqDiffMat.sum()
    print 'sortedDistIndicies:', sortedDistIndicies
    print 'classCount.iteritems():', classCount.iteritems()
    print 'classCount:', classCount
    print 'sortedClassCount:', sortedClassCount

    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t') # split line with tab
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('sources/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], \
                                     datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" \
            % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input(\
                  "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('sources/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print "You will probably like this persion: ", resultList[classifierResult - 1]


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest(trainDir, testDir):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(trainDir)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(trainDir+fileNameStr)
    testFileList = listdir(testDir)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(testDir+fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" \
            % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))



run = 5
if run == 1:
    group, labels = createDataSet()
    classify0_debug([1, 0.1], group, labels, 3)
elif run == 2: # Display the data set and try to classify it
    import matplotlib
    import matplotlib.pyplot as plt

    datingDataMat, datingLabels = file2matrix('sources/Ch02/datingTestSet2.txt')
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    ax1.set_xlabel('Percentage of Time Spent Playing Video Games')
    ax1.set_ylabel('Liters of Ice Cream Consumed Per Week')

    ax2 = fig.add_subplot(212)
    ax2.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
    ax2.set_xlabel('Frequent Flyier Miles Earned Per Year')
    ax2.set_ylabel('Percentage of Time Spent Playing Video Games')
    plt.show()
elif run == 3:
    datingClassTest()
elif run == 4:
    classifyPerson()
elif run == 5:
    testDir = '/Users/austin/work/temp/digits/testDigits/'
    trainDir = '/Users/austin/work/temp/digits/trainingDigits/'
    handwritingClassTest(trainDir, testDir)