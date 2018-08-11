#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import timeit
from PIL import Image
minlabel=54.0
maxlabel=82781.0
featureId_maxVal ={1:159.78,2:24879.09,3:122.82,4:144.0,5:112.01,6:144.0,7:120.55,8:120.55,9:121.17,10:126.0,11:120.39,12:144.0,13:126.0,14:126.0,15:121.5,16:121.5,17:120.35,18:126.0,19:126.0,20:126.0,21:120.49,22:120.54,23:120.71,24:120.6,25:126.0,26:144.0,27:144.0,28:144.0,29:144.0,30:144.0,31:144.0,32:144.0,33:144.0,34:144.0,35:144.0,36:144.0,37:144.0,38:144.0,39:144.0,40:144.0,41:144.0,42:144.0,43:144.0,44:122.82,45:122.82,46:120.67,47:123.0,48:126.0,49:124.36,50:126.0,51:122.4,52:126.0,53:120.68,54:121.09,55:121.09,56:122.82,57:144.0,58:144.0,59:144.0,60:144.0,61:144.0,62:144.0,63:144.0,64:144.0,65:144.0,66:144.0,67:144.0,68:144.0,69:144.0,70:144.0,71:144.0,72:144.0,73:144.0,74:144.0,75:144.0,76:144.0,77:122.4,78:122.4,79:122.4,80:144.0,81:144.0,82:144.0,83:144.0,84:144.0,85:144.0,86:144.0,87:144.0,88:144.0,89:123.43,90:126.0,91:121.33,92:120.96,93:123.0,94:122.4,95:123.43,96:121.37,97:121.5,98:123.43,99:123.43,100:122.4,101:120.21,102:121.33,103:144.0,104:120.1,105:144.0,106:1.0,107:1.0,108:1.0,109:1.0,110:1.0,111:0.0,112:1.0,113:0.0,114:1.0,115:0.91,116:0.0,117:0.79,118:184543.0,119:7.0,120:0.0,121:14.0,122:0.0,123:4.0,124:3.0,125:4.0,126:9.0,127:2.0,128:0.0,129:0.0,130:3.0,131:8.0,132:2.0,133:0.0,134:3.0,135:0.0,136:0.0,137:4.0,138:0.0,139:0.0,140:0.0,141:1.0,142:0.0,143:0.0,144:0.0,145:0.0,146:0.0,147:0.0,148:0.0,149:0.0,150:0.0,151:0.0,152:0.0,153:0.0,154:0.0,155:0.0,156:0.0,157:0.0,158:1.0,159:0.0,160:0.0,161:0.0,162:0.0,163:4.0,164:0.0,165:0.0,166:0.0,167:0.0,168:2.0,169:1.0,170:1.0,171:0.0,172:5.0,173:4.0,174:2.0,175:0.0,176:1.0,177:1.0,178:0.0,179:1.0,180:7.0,181:6.0,182:0.0,183:4.0,184:2.0,185:0.0,186:4.0,187:4.0,188:2.0,189:0.0,190:5.0,191:0.0,192:4.0,193:3.0,194:2.0,195:0.0,196:2.0,197:0.0,198:0.0,199:1.0,200:1.0,201:5.0,202:4.0,203:4.0,204:0.0,205:6.0,206:2.0,207:0.0,208:4.0,209:3.0,210:4.0,211:2.0,212:0.0,213:10.0,214:0.0,215:4.0,216:2.0,217:9.0,218:3.0,219:2.0,220:0.0,221:3.0,222:2.0,223:6.0,224:0.0,225:7.0,226:0.0,227:5.0,228:3.0,229:8.0,230:0.0,231:0.0,232:9.0,233:13.0,234:7.0,235:3.0,236:6.0,237:0.0,238:0.0,239:0.0,240:2.0,241:4.0,242:0.0,243:0.0,244:2.0,245:0.0,246:0.0,247:0.0,248:2.0,249:0.0,250:0.0,251:7.0,252:6.0,253:8.0,254:116}
featureId_minVal = {1:1.6,2:0.0,3:0.0,4:1.5,5:1.5,6:0.0,7:0.0,8:0.0,9:0.0,10:0.0,11:0.0,12:0.0,13:0.0,14:0.0,15:0.0,16:0.0,17:0.0,18:0.0,19:0.0,20:0.0,21:0.0,22:0.0,23:0.0,24:0.0,25:0.0,26:0.0,27:0.0,28:0.0,29:0.0,30:0.0,31:0.0,32:0.0,33:0.0,34:0.0,35:0.0,36:0.0,37:0.0,38:0.0,39:0.0,40:0.0,41:0.0,42:0.0,43:0.0,44:0.0,45:0.0,46:0.0,47:0.0,48:0.0,49:0.0,50:0.0,51:0.0,52:0.0,53:0.0,54:0.0,55:0.0,56:0.0,57:0.0,58:0.0,59:0.0,60:0.0,61:0.0,62:0.0,63:0.0,64:0.0,65:0.0,66:0.0,67:0.0,68:0.0,69:0.0,70:0.0,71:0.0,72:0.0,73:0.0,74:0.0,75:0.0,76:0.0,77:0.0,78:0.0,79:0.0,80:0.0,81:0.0,82:0.0,83:0.0,84:0.0,85:0.0,86:0.0,87:0.0,88:0.0,89:0.0,90:0.0,91:0.0,92:0.0,93:0.0,94:0.0,95:0.0,96:0.0,97:0.0,98:1.5,99:1.5,100:1.5,101:1.5,102:1.5,103:1.5,104:1.5,105:1.5,106:0.0,107:0.0,108:0.0,109:0.0,110:0.0,111:0.0,112:0.0,113:0.0,114:0.0,115:0.0,116:0.0,117:0.0,118:2001.0,119:0.0,120:0.0,121:0.0,122:0.0,123:0.0,124:0.0,125:0.0,126:0.0,127:0.0,128:0.0,129:0.0,130:0.0,131:0.0,132:0.0,133:0.0,134:0.0,135:0.0,136:0.0,137:0.0,138:0.0,139:0.0,140:0.0,141:0.0,142:0.0,143:0.0,144:0.0,145:0.0,146:0.0,147:0.0,148:0.0,149:0.0,150:0.0,151:0.0,152:0.0,153:0.0,154:0.0,155:0.0,156:0.0,157:0.0,158:0.0,159:0.0,160:0.0,161:0.0,162:0.0,163:0.0,164:0.0,165:0.0,166:0.0,167:0.0,168:0.0,169:0.0,170:0.0,171:0.0,172:0.0,173:0.0,174:0.0,175:0.0,176:0.0,177:0.0,178:0.0,179:0.0,180:0.0,181:0.0,182:0.0,183:0.0,184:0.0,185:0.0,186:0.0,187:0.0,188:0.0,189:0.0,190:0.0,191:0.0,192:0.0,193:0.0,194:0.0,195:0.0,196:0.0,197:0.0,198:0.0,199:0.0,200:0.0,201:0.0,202:0.0,203:0.0,204:0.0,205:0.0,206:0.0,207:0.0,208:0.0,209:0.0,210:0.0,211:0.0,212:0.0,213:0.0,214:0.0,215:0.0,216:0.0,217:0.0,218:0.0,219:0.0,220:0.0,221:0.0,222:0.0,223:0.0,224:0.0,225:0.0,226:0.0,227:0.0,228:0.0,229:0.0,230:0.0,231:0.0,232:0.0,233:0.0,234:0.0,235:0.0,236:0.0,237:0.0,238:0.0,239:0.0,240:0.0,241:0.0,242:0.0,243:0.0,244:0.0,245:0.0,246:0.0,247:0.0,248:0.0,249:0.0,250:0.0,251:0.0,252:0.0,253:0.0,254:0.0}
'''将特征集转化成tfrecord格式数据，其中特征集使用MR跑的文本文件'''
def getTfrecordFile(inputFilePath,test_outfilePath,train_outfilePath,percent=10):
    '''
    参考文献：https://blog.csdn.net/xierhacker/article/details/72357651
    https://blog.csdn.net/waterydd/article/details/72866109
    :param inputFilePath:输入文件目录
    :param outfilePath:输出文件目录
    :return:
    '''
    test_writer = tf.python_io.TFRecordWriter(test_outfilePath)
    train_writer = tf.python_io.TFRecordWriter(train_outfilePath)
    start_time = timeit.default_timer()
    totalNum=0
    n=0
    #featuresList = []
    with open(inputFilePath, "r") as file:
        for line in file:
            featuresList = []
            arr1=line.split(",")
            arr = []
            #print(arr1)
            arr= arr1[0].split(" ")
            #print(arr)
            totalNum = totalNum + 1
            if totalNum % 1000 == 0:
                print(totalNum)
            okflag=True
            arrlen=len(arr)
            label = 0
            for i in arr[0:1]:
                label = int(float(i.split(":")[2]))
                # label = float((label-minlabel)/(maxlabel-minlabel))
                # if label>1 or label<0:
                #     print("youcuowu1")
                #     okflag = False

            for j in range(1,arrlen):
                mm=str(arr[j]).split(":")
                featureId=mm[0]
                featureVal=mm[1].strip()
                if j == 1:
                    if float(featureVal)>160:
                        okflag=False
                if j == 4:
                    if float(featureVal)>160:
                        okflag=False

                # if j in range(106,117):
                #     #print(float(featureVal))
                #     featuresList.append((float(featureVal)))
                #     continue
                # if j == 254:
                #     print(mm[1])

                if j > 254:
                    okflag = False

                # fenmu = (float(featureId_maxVal[j])-float(featureId_minVal[j]))
                #
                #
                # if fenmu!=0:
                #     pc = (float(mm[1])-float(featureId_minVal[j]))/fenmu
                # else:
                #     pc = 0
                #
                # if pc > 1 or pc < 0:
                #     okflag = False
                featuresList.append((float(featureVal)))
            #print(featuresList)
            dataMat = featuresList
            r_n_eigVect = np.load("n_eigVect_new.npy")
            meanVal = np.load("meanVal.npy")
            stdDeviation = np.load("stdDeviation.npy")

            newData = (dataMat - meanVal) / (stdDeviation + 0.000000001)  # 方差归一化，防止分母为0

            lowDDataMat = np.dot(newData, r_n_eigVect)
            #print(lowDDataMat.real)
            featuresList0 = lowDDataMat
            #nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #print("end:", nowTime)
            #print(featuresList)
            # for i in range(6):
            #     featuresList.append(int(-1))
            # 将数据转化为原生 bytes
            if okflag==False:
                continue
            #features1=np.reshape(featuresList,(16,16))
            # image = Image.fromarray(features1)
            features1 = np.array(featuresList0)
            #print(features1)
            #print(len(features1))
            image_bytes = features1.tobytes()
            # 创建字典
            features = {}
            # 用bytes来存储image
            #print(label)
            features['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            # 用int64来表达label
            # features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            # print(label)
            features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))

            # 将所有的feature合成features
            tf_features = tf.train.Features(feature=features)
            # 转成example
            tf_example = tf.train.Example(features=tf_features)
            # 序列化样本
            tf_serialized = tf_example.SerializeToString()
            # example = tf.train.Example(features=tf.train.Features(feature={
            #     "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            #     "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[features2])),
            # }))
            if totalNum%percent==0:
                n = n + 1
                # print(n)
                test_writer.write(tf_serialized)
            else:
                train_writer.write(tf_serialized)  # 序列化为字符串
    train_writer.close()
    test_writer.close()

    end_time = timeit.default_timer()
    print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))
if __name__=="__main__":
    getTfrecordFile("ETAFeature2GBDT_20180516.csv","./eta_test.tfrecords","./eta_train.tfrecords")


