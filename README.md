# ETA
Estimated time of arrival 
ETA，即ESTIMATED TIME OF ARRIVAL 中文为“预计到达时间”，是出行服务中的一个重要指标。是指在电子地图中，对某一条给定的路线，结合实时特征
（如实时路况、天气、交通事故、时间等）、静态特征（如路网信息、POI信息），甚至还有对未来信息的预计（未来几小时天气状况预估、未来几小时路
况变化预估、交通事故拥塞变化预计等），计算出通过不同出行方式（如驾车、骑行、不行、公共交通等）所需要花费的时间，而其中最为复杂、目前最不
精准的，就是驾车方式下的ETA估计。
PCA文件即利用PCA将特征降维到一定的维数来加快训练速度，这是由于硬件限制多做的步骤
CSVtoTFrecord文件是将数据转换为tensorflow特有的数据转换格式TFrecord来加快文件读写速度
剩下的三组文件即是利用CNN、DNN来进行ETA训练测试的步骤，
测试数据是训练数据的10%，共29万+的测试数据，在DNN2.0中准确率：有41.9%误差在10%以下，29.6%误差在10%-20%，30%以内的占比为83.8%。