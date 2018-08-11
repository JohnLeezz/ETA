#！/usr/bin/python
#-*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError, InvalidArgumentError
import time
'''
DNN
'''
class regression_dnn(object):
    def __init__(self, test_dataset, train_dataset,modelPath,testLogFilePath):
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.modelPath=modelPath
        self.testLogFilePath=testLogFilePath
        self.creat()

    def read_and_decode(self,example_proto):
        # 定义解析的字典
        dics = {}
        dics['label'] = tf.FixedLenFeature(shape=[], dtype=tf.float32)
        dics['image'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
        # 调用接口解析一行样本
        parsed_example = tf.parse_single_example(serialized=example_proto, features=dics)
        image = tf.decode_raw(parsed_example['image'], out_type=tf.float64)
        # image=parsed_example['image']
        # image = tf.cast(image, tf.int64)
        #image = tf.reshape(image, shape=[16 * 16])
        # 这里对图像数据做归一化，是关键，没有这句话，精度不收敛，为0.1左右，
        # 有了这里的归一化处理，精度与原始数据一致
        label = parsed_example['label']
        label = tf.cast(label, tf.float32)
        # label = tf.one_hot(label, depth=1, on_value=1)
        # label = tf.image.convert_image_dtype(label, tf.string)
        # label = tf.one_hot(label, depth=1, on_value=1)
        return image, label

    def compute_cost(pre, real):
        pass

    def creat(self):
        with tf.variable_scope("rnn") as scope1:
            xs = tf.placeholder(tf.float32, [None, 137])
            ys = tf.placeholder(tf.float32, [None, 1])

            keep_prob = tf.placeholder(tf.float32)

            tf.set_random_seed(1)

            w_1 = tf.get_variable("w_1", shape=[137, 512], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b_1 = tf.get_variable("b_1", shape=[1, 512], initializer=tf.zeros_initializer())
            z_1 = tf.add(tf.matmul(xs, w_1), b_1)
            a_1_drop = tf.nn.dropout(z_1, keep_prob)
            a_1 = tf.nn.relu(a_1_drop)

            w_2 = tf.get_variable("w_2", shape = [512, 768], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b_2 = tf.get_variable("b_2", shape=[1, 768], initializer=tf.zeros_initializer())
            z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
            a_2_drop = tf.nn.dropout(z_2, keep_prob)
            a_2 = tf.nn.relu(a_2_drop)

            w_3 = tf.get_variable("w_3", shape = [768, 348], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b_3 = tf.get_variable("b_3", shape=[1, 348], initializer=tf.zeros_initializer())
            z_3 = tf.add(tf.matmul(a_2, w_3), b_3)
            a_3_drop = tf.nn.dropout(z_3, keep_prob)
            a_3 = tf.nn.relu(a_3_drop)

            w_4 = tf.get_variable("w_4", shape = [348, 128], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b_4 = tf.get_variable("b_4", shape=[1, 128], initializer=tf.zeros_initializer())
            z_4 = tf.add(tf.matmul(a_3, w_4), b_4)
            a_4_drop = tf.nn.dropout(z_4, keep_prob)
            a_4 = tf.nn.relu(a_4_drop)

            w_5 = tf.get_variable("w_5", shape = [128, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b_5 = tf.get_variable("b_5", shape=[1, 1], initializer=tf.zeros_initializer())
            z_5 = tf.add(tf.matmul(a_4, w_5), b_5)

            # w_1 = tf.get_variable("w_1", shape=[137, 512], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            # b_1 = tf.get_variable("b_1", shape=[1, 512], initializer=tf.zeros_initializer())
            # z_1 = tf.add(tf.matmul(xs, w_1), b_1)
            # # a_1_drop = tf.nn.dropout(z_1, keep_prob)
            # a_1 = tf.nn.relu(z_1)
            #
            # w_2 = tf.get_variable("w_2", shape=[512, 1024], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            # b_2 = tf.get_variable("b_2", shape=[1, 1024], initializer=tf.zeros_initializer())
            # z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
            # # a_2_drop = tf.nn.dropout(z_2, keep_prob)
            # a_2 = tf.nn.relu(z_2)
            #
            # w_3 = tf.get_variable("w_3", shape=[1024, 348], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            # b_3 = tf.get_variable("b_3", shape=[1, 348], initializer=tf.zeros_initializer())
            # z_3 = tf.add(tf.matmul(a_2, w_3), b_3)
            # # a_3_drop = tf.nn.dropout(z_3, keep_prob)
            # a_3 = tf.nn.relu(z_3)
            #
            # w_4 = tf.get_variable("w_4", shape=[348, 128], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            # b_4 = tf.get_variable("b_4", shape=[1, 128], initializer=tf.zeros_initializer())
            # z_4 = tf.add(tf.matmul(a_3, w_4), b_4)
            # a_4_drop = tf.nn.dropout(z_4, keep_prob)
            # a_4 = tf.nn.relu(a_4_drop)
            #
            # w_5 = tf.get_variable("w_5", shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            # b_5 = tf.get_variable("b_5", shape=[1, 1], initializer=tf.zeros_initializer())
            # z_5 = tf.add(tf.matmul(a_4, w_5), b_5)

            pre = tf.nn.relu(z_5)
            # cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pre), reduction_indices=[1]))
            # #指数衰减法设置学习率
            # global_step = tf.Variable(0)
            # learning_rate = tf.train.exponential_decay(0.001, global_step, 50, 0.98, staircase=True)
            # # 0.01学习效率,minimize(loss)减小loss误差
            # train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)


            with tf.variable_scope("readTrainData")as testData_scope:
                dataset = tf.data.TFRecordDataset(filenames=[self.test_dataset])
                dataset = dataset.map(self.read_and_decode)
                dataset = dataset.batch(1).repeat(1)
                iterator = dataset.make_one_shot_iterator()
                next_element = iterator.get_next()

            model_path =self.modelPath
            saver = tf.train.Saver(max_to_keep=1)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, model_path)
                start_time = time.time()
                i = 0
                # ss = preprocessing.StandardScaler()
                while True:
                    i = i + 1
                    try:
                        image, label = sess.run(fetches=next_element)
                        # print(label)
                        # image = ss.fit_transform(image)
                        # print(image)
                        label = label.reshape(-1, 1)
                        # print(sess.run(prediction, feed_dict={xs: image,  keep_prob: 1.0}))
                        # print(label)
                    except OutOfRangeError as e:
                        print('OutOfRangeError', e)
                        break
                    except TypeError as e1:
                        print("TypeError:", e1)
                        continue
                    except InvalidArgumentError as e2:
                        print("TypeError:", e2)
                        continue
                    else:
                        # sess.run(train_step, feed_dict={xs: image, ys: label, keep_prob: 0.5})
                        if i % 1 == 0:
                            label = label.reshape(-1, 1)
                            predictVal = sess.run(pre, {xs: image, ys: label, keep_prob: 1.0})
                            # print(predictVal*68000)
                            print(abs(label - predictVal) / label)
                            with open(self.testLogFilePath, "a") as o:
                                o.write(str(abs(label - predictVal) / label).replace("[[", "").replace("]]", "")+","+str(label).replace("[[", "").replace("]]", "")+","+str(predictVal).replace("[[", "").replace("]]", ""))
                                o.write("\n")

                end_time = time.time()
                print("************************************")
                print("Time Cost:", (end_time - start_time))

# obj = regression_dnn("eta_test_137.tfrecords", "eta_train_137.tfrecords","etaModel_DNN/model.ckpt")
obj = regression_dnn("eta_test_137.tfrecords", "eta_train_137.tfrecords","etaModel_DNN_v1.0/model.ckpt","testLogFilePath_1.0.log")



