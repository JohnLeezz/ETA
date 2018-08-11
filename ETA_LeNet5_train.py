#加正则化项防止过拟合
#自动梯度下降学习率
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError, InvalidArgumentError
import time as time
class regression_cnn(object):
    def __init__(self,testDataSet,train_dataSet,outFile):
        self.testDataSet=testDataSet
        self.train_dataSet = train_dataSet
        self.outFile = outFile
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
        image = tf.reshape(image, shape=[16 * 16])
        # 这里对图像数据做归一化，是关键，没有这句话，精度不收敛，为0.1左右，
        # 有了这里的归一化处理，精度与原始数据一致
        label = parsed_example['label']
        label = tf.cast(label, tf.float32)
        # label = tf.one_hot(label, depth=1, on_value=1)
        # label = tf.image.convert_image_dtype(label, tf.string)
        # label = tf.one_hot(label, depth=1, on_value=1)
        return image, label

    def creat(self):
        with tf.variable_scope("cnn") as scope1:
            # define placeholder for inputs to network
            xs = tf.placeholder(tf.float32, [None, 256])  # 原始数据的维度：16
            ys = tf.placeholder(tf.float32, [None, 1])  # 输出数据为维度：1

            #keep_prob = tf.placeholder(tf.float32)  # dropout的比例

            x_image = tf.reshape(xs, [-1, 16, 16, 1])  # 原始数据16变成二维图片4*4
            # 第一层：卷积层
            conv1_weights = tf.get_variable("conv1_weights", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(
                stddev=0.1))  # 过滤器大小为5*5, 当前层深度为1， 过滤器的深度为32
            conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')  # 移动步长为1, 使用全0填充
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))  # 激活函数Relu去线性化
            # 第二层：最大池化层
            # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 第三层：卷积层
            conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64],
                                            initializer=tf.truncated_normal_initializer(
                                                stddev=0.1))  # 过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64
            conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')  # 移动步长为1, 使用全0填充
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
            # 第四层：最大池化层
            # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 第五层：全连接层
            fc1_weights = tf.get_variable("fc1_weights", [4 * 4 * 64, 512],
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=0.1))  # 7*7*64=3136把前一层的输出变成特征向量
            fc1_baises = tf.get_variable("fc1_baises", [512], initializer=tf.constant_initializer(0.1))
            pool2_vector = tf.reshape(pool2, [-1, 4 * 4 * 64])
            fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)
            # 为了减少过拟合，加入Dropout层
            keep_prob = tf.placeholder(tf.float32)
            fc1_dropout = tf.nn.dropout(fc1, keep_prob)
            # 第六层：全连接层
            fc2_weights = tf.get_variable("fc2_weights", [512, 1], initializer=tf.truncated_normal_initializer(
                stddev=0.1))  # 神经元节点数1024, 分类节点10
            fc2_biases = tf.get_variable("fc2_biases", [1], initializer=tf.constant_initializer(0.1))
            prediction = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases


            # 计算 predition与y 差距 所用方法很简单就是用 suare()平方,sum()求和,mean()平均值
            cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
            # 0.01学习效率,minimize(loss)减小loss误差
            # 指数衰减法设置学习率
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(0.0001, global_step, 2390, 0.98, staircase=True)
            # 0.01学习效率,minimize(loss)减小loss误差
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy,
                                                                                      global_step=global_step)
            #train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

            with tf.variable_scope("readTrainData")as testData_scope:
                dataset = tf.data.TFRecordDataset(filenames=['eta_train.tfrecords'])
                dataset = dataset.map(self.read_and_decode)
                dataset = dataset.batch(1).repeat(1000)
                iterator = dataset.make_one_shot_iterator()
                next_element = iterator.get_next()

            saver = tf.train.Saver()
            model_path = './model_LeNet5/model.ckpt'
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess,model_path)

                i=0
                #ss = preprocessing.StandardScaler()
                while True:
                    i = i + 1
                    try:
                        image, label = sess.run(fetches=next_element)
                        #print(label)
                        #image = ss.fit_transform(image)
                        #print(image)
                        label = label.reshape(-1, 1)
                        # print(sess.run(prediction, feed_dict={xs: image,  keep_prob: 1.0}))
                        #print(label)
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
                        sess.run(train_step, feed_dict={xs: image, ys: label, keep_prob: 0.5})
                        saver.save(sess, model_path)
                        if i % 50 == 0:
                            print(i,time.strftime("%Y%m%d%H%M", time.localtime()), '误差=', sess.run(cross_entropy, feed_dict={xs: image, ys: label,

                                                                       keep_prob: 1.0}))  # 输出loss值

obj=regression_cnn("eta_test.tfrecords","eta_train.tfrecords","./model/model.ckpt")





