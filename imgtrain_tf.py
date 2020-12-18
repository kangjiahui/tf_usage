import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time
from sklearn.model_selection import train_test_split
from skimage import io  # multiple images
import skimage.transform
import warnings


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def resize(f):
    rgb = Image.open(f).convert("RGB")
    rgb = np.array(rgb)
    rgb = rgb.astype(np.float64)
    rgb -= [123.68, 116.779, 103.939]
    bgr = rgb[..., ::-1]
    resized = skimage.transform.resize(bgr, (32, 32))
    return resized


img_width = 32
img_height = 32
n_input = img_width * img_height
n_output = 10

path_train = []
for i in range(10):
    path_train.append('image/' + str(i) + '/*')
print(path_train)
coll_train = io.ImageCollection(path_train, load_func=resize)
mat_train = io.concatenate_images(coll_train)
x_train = mat_train.reshape(-1, img_width, img_height, 3)
print(x_train.shape)
y_train = np.zeros((x_train.shape[0], n_output))
print(y_train.shape)
for j in range(10):
    for i in range(x_train.shape[0]):
        if (i >= j * 88) and (i < (j + 1) * 88):
            y_train[i][j] = 1.0

# 默认shuffle=True
# random_state：设置随机数种子，保证每次都是同一个随机数。若为0或不填，则每次得到数据都不一样
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
print(X_Train.shape, Y_Train.shape)
print(X_Test.shape, Y_Test.shape)
print('Data set Complete!')


def model1(_input, _w, _b, _p_keep_conv, _p_keep_hidden):
    # _input: must be (-1, 128, 128, 1) shape
    # _conv1 (-1, 64, 64, 64)
    _conv1 = tf.nn.conv2d(_input, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    _conv1_r = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    # _pool1 (-1, 32, 32, 64)
    _pool1 = tf.nn.max_pool(_conv1_r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool1_drp = tf.nn.dropout(_pool1, p_keep_conv)

    # _conv2 (-1, 32, 32, 128)
    _conv2 = tf.nn.conv2d(_pool1_drp, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    # _pool2 (-1, 16, 16, 128)
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool2_drp = tf.nn.dropout(_pool2, p_keep_conv)

    _dense1 = tf.reshape(_pool2_drp, [-1, _w['wd1'].get_shape().as_list()[0]])
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc1_drp = tf.nn.dropout(_fc1, p_keep_hidden)

    _out = tf.add(tf.matmul(_fc1_drp, _w['wd2']), _b['bd2'])

    out = {
        'fc1': _fc1,
        'out': _out
    }
    return out


with tf.device('/gpu:1'):
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 3, 8], stddev=0.1)),
        'wc2': tf.Variable(tf.random_normal([3, 3, 8, 16], stddev=0.1)),
        'wd1': tf.Variable(tf.random_normal([8 * 8 * 16, 64], stddev=0.1)),
        'wd2': tf.Variable(tf.random_normal([64, n_output], stddev=0.1)),
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([8], stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([16], stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([64], stddev=0.1)),
        'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1)),
    }

x = tf.placeholder(tf.float32, [None, img_width, img_height, 3])
y = tf.placeholder(tf.float32, [None, n_output])
p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

output = model1(x, weights, biases, p_keep_conv, p_keep_hidden)
predict = output['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
training_epochs = 200

# SAVER
save_step = 2
saver = tf.train.Saver(max_to_keep=2)
data_dir = "./datadir"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# actcode
# = 0, no read any data, epoch start from 1 to training_epochs. Save net para
# = 1, resume training, read data from file #data_index, epoch start from #epo_start to training_epochs. Save net para
# = 2, read data from file #data_index, no training, only inspect correction percentage
actcode = 0
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(init)

batch_size = 50
display_step = 1  # number of epoch print result

if actcode == 1 or actcode == 2:
    data_index = 300
    saver.restore(sess, data_dir + "/myfirst.ckpt_" + str(data_index))

if actcode == 0 or actcode == 1:
    for epoch in range(training_epochs):
        time_start = time.time()
        # average cost of each batch
        avg_cost = 0.0
        training_batch = zip(range(0, len(X_Train), batch_size), range(batch_size, len(X_Train) + 1, batch_size))
        batch_num = 0
        for start, end in training_batch:
            batch_num = batch_num + 1
            sess.run(optimizer,
                     feed_dict={x: X_Train[start:end], y: Y_Train[start:end], p_keep_conv: 0.9, p_keep_hidden: 0.9})
            avg_cost += sess.run(cost, feed_dict={x: X_Train[start:end], y: Y_Train[start:end], p_keep_conv: 1.0,
                                                  p_keep_hidden: 1.0})

        # Training result output per epoch
        avg_cost = avg_cost / batch_num
        acc_train = sess.run(accuracy, feed_dict={x: X_Train, y: Y_Train, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        acc_test = sess.run(accuracy, feed_dict={x: X_Test, y: Y_Test, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        time_end = time.time()
        print("epoch: %03d" % (epoch+1))
        print("Train time cost: %.6f" % (time_end - time_start))
        print("Train cost average: %.6f" % avg_cost)
        print("Train set accuracy: %.6f" % acc_train)
        print("Test set accuracy: %.6f" % acc_test)

        # Save Net
        if epoch % save_step == 0:
            saver.save(sess, data_dir + "/myfirst.ckpt_" + str(epoch))

if actcode == 2:

    path_test = 'image/0/*.png'
    coll_test = io.ImageCollection(path_test, load_func=resize)
    mat_test = io.concatenate_images(coll_test)
    x_test = mat_test.reshape(-1, img_width, img_height, 1)  # size (26*900, 128, 128, 1)
    print(x_test.shape)
    y_test = np.zeros((x_test.shape[0], n_output))
    for i in range(0, x_test.shape[0]):
        y_test[i][0] = 1.0

    acc_test = sess.run(accuracy, feed_dict={x: x_test, y: y_test, p_keep_conv: 1.0, p_keep_hidden: 1.0})
    print("Test set accuracy: %.6f" % acc_test)
