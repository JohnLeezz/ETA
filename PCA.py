from numpy import random
import numpy as np
import datetime
from sklearn import preprocessing
def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    return newData,meanVal
def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num
def pca(dataMat,percentage=0.99):
    # newData,meanVal=zeroMean(dataMat)
    meanVal = np.mean(dataMat, axis=0)

    # newData = preprocessing.scale(dataMat)#归一化，期望=0，方差=1

    meanVal=np.mean(dataMat,axis=0)#numpy实现 归一化，期望=0，方差=1
    stdDeviation=np.std(dataMat,axis=0)
    newData=(dataMat - meanVal) / (stdDeviation+0.000000001)#防止分母为0
    print(newData)

    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    print("eigVals:",eigVals)
    n=percentage2n(eigVals,percentage)
    print(n)
    # print(newData.shape)#要达到percent的方差百分比，需要前n个特征向量
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    # print(eigValIndice)
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]      #最大的n个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect
    print("aaa",lowDDataMat)

    #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据
    return lowDDataMat,reconMat,n_eigVect,meanVal,stdDeviation


def formatConversion(inputFile, outputFile):
    '''
    转换数据格式
    :param inputFile:
    :param outputFile:
    :return:
    '''
    writeFile = open(outputFile, "w")
    with open(inputFile,"r") as file:
        for line in file:
            outLine=""
            arr=line.split(" ")
            arrLen=len(arr)
            for i in range(1,arrLen):
                nowFeature=arr[i].split(":")[1]
                outLine+=nowFeature
                if i!=arrLen-1:
                    outLine+=" "
            # print(outLine)
            writeFile.write(outLine)
    writeFile.close()

if __name__=="__main__":

    # formatConversion(r"D:\data\ETAFeature2GBDT_201805161\ETAFeature2GBDT_201805161.csv",r"D:\data\ETAFeature2GBDT_201805161\ETAFeature2GBDT_201805161_new.csv")
    #-----------------------------------训练PCA-----------------------------------------#
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("start:", nowTime)
    filePath = "d:/data/20170404_3/array.txt"
    dataMat = np.loadtxt(filePath, delimiter=" ")
    lowDDataMat, reconMat,n_eigVect,meanVal,stdDeviation=pca(dataMat)
    # -----------------------------------保存参数---------------------------------------#
    newData=preprocessing.scale(dataMat)#归一化，均值0，方差1
    np.save("n_eigVect.npy",n_eigVect)#保存特征向量
    np.save("meanVal.npy", meanVal)#保存期望值
    np.save("stdDeviation.npy", stdDeviation)#保存标准差值
    #-----------------------------------应用--------------------------------------------#
    r_n_eigVect=np.load("n_eigVect.npy")
    meanVal = np.load("meanVal.npy")
    stdDeviation = np.load("stdDeviation.npy")

    newData = (dataMat - meanVal) / (stdDeviation + 0.000000001)  # 方差归一化，防止分母为0

    lowDDataMat =np.dot(newData,r_n_eigVect)
    print(lowDDataMat.real)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("end:", nowTime)