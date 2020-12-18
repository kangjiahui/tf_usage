import tensorflow as tf
import numpy as np
from skimage import color, transform


class ModuelTF(object):
    def __init__(self, input_size=[32, 32], n_output=10, path="./datadir", data_index=1000, threshold=0.7):
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.input_size = input_size
        self.n_output = n_output
        self.path = path
        self.data_index = data_index
        self.threshold = threshold
        self.init = tf.global_variables_initializer()
        filepath = self.path + "/myfirst.ckpt_" + str(self.data_index)
        self.output, self.x, self.p_keep_conv, self.p_keep_hidden = self.pre_work()
        self.result = tf.argmax(self.output, 1)
        saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        tf.reset_default_graph()
        saver.restore(self.sess, filepath)

    def model1(self, _input, _w, _b, p_keep_conv, p_keep_hidden):
        # _input: must be (-1, 16, 16, 1) shape
        _conv1 = tf.nn.conv2d(_input, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        # _conv1 (-1, 16, 16, 8)
        _conv1_r = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
        _pool1 = tf.nn.max_pool(_conv1_r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # _pool1 (-1, 8, 8, 8)
        _pool1_drp = tf.nn.dropout(_pool1, p_keep_conv)
        _conv2 = tf.nn.conv2d(_pool1_drp, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        # _conv2 (-1, 8, 8, 16)
        _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
        _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # _pool2 (-1, 4, 4, 16)
        _pool2_drp = tf.nn.dropout(_pool2, p_keep_conv)
        _dense1 = tf.reshape(_pool2_drp, [-1, _w['wd1'].get_shape().as_list()[0]])
        _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
        _fc1_drp = tf.nn.dropout(_fc1, p_keep_hidden)
        _out = tf.add(tf.matmul(_fc1_drp, _w['wd2']), _b['bd2'])
        return _out

    def forward(self, pic):
        pic = np.array(pic)
        pic = pic.astype(np.float64)
        resized = transform.resize(pic, (self.input_size[0], self.input_size[1]))
        resized = resized.reshape(1, self.input_size[0], self.input_size[1], 3)

        label_index, pred = self.sess.run([self.result, self.output],
                                    feed_dict={self.x: resized, self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0})
        return label_index, pred

    def pre_work(self):
        with tf.device('/gpu:1'):
            weights = {
                'wc1': tf.Variable(tf.random_normal([3, 3, 3, 8], stddev=0.1)),
                'wc2': tf.Variable(tf.random_normal([3, 3, 8, 16], stddev=0.1)),
                'wd1': tf.Variable(tf.random_normal([8 * 8 * 16, 64], stddev=0.1)),
                'wd2': tf.Variable(tf.random_normal([64, self.n_output], stddev=0.1)),
            }

            biases = {
                'bc1': tf.Variable(tf.random_normal([8], stddev=0.1)),
                'bc2': tf.Variable(tf.random_normal([16], stddev=0.1)),
                'bd1': tf.Variable(tf.random_normal([64], stddev=0.1)),
                'bd2': tf.Variable(tf.random_normal([self.n_output], stddev=0.1)),
            }
        x = tf.placeholder(tf.float32, [None, self.input_size[0], self.input_size[1], 3])
        p_keep_conv = tf.placeholder(tf.float32)
        p_keep_hidden = tf.placeholder(tf.float32)

        output = self.model1(x, weights, biases, p_keep_conv, p_keep_hidden)
        return output, x, p_keep_conv, p_keep_hidden

    def get_result(self, pic):
        index, pred = self.forward(pic)
        label = self.classes[index[0]]
        z = zip(label, pred)
        label_new = ""
        for b, c in z:
            prob = max(np.exp(c) / sum(np.exp(c)))
            print(prob)
            if prob < float(self.threshold):
                label_new = "NULL"
            else:
                label_new = b
        return label_new

    def close_session(self):
        self.sess.close()
