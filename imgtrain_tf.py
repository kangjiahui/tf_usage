import tensorflow as tf
import numpy as np
import os
from PIL import Image
import time
from sklearn.model_selection import train_test_split
from skimage import io, color  # multiple images
import skimage.transform


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
    _conv1 = tf.nn.conv2d(_input, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    # _conv1 (-1, 64, 64, 64)
    _conv1_r = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1_r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # _pool1 (-1, 32, 32, 64)
    _pool1_drp = tf.nn.dropout(_pool1, p_keep_conv)

    _conv2 = tf.nn.conv2d(_pool1_drp, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    # _conv2 (-1, 32, 32, 128)
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # _pool2 (-1, 16, 16, 128)
    _pool2_drp = tf.nn.dropout(_pool2, p_keep_conv)

    _dense1 = tf.reshape(_pool2_drp, [-1, _w['wd1'].get_shape().as_list()[0]])
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc1_drp = tf.nn.dropout(_fc1, p_keep_hidden)

    _out = tf.add(tf.matmul(_fc1_drp, _w['wd2']), _b['bd2'])

    out = {
        'inputx': _input,
        'conv1': _conv1,
        'conv1_r': _conv1_r,
        'pool1': _pool1,
        'pool1_drp': _pool1_drp,
        'conv2': _conv2,
        'pool2': _pool2,
        'pool2_drp': _pool2_drp,
        'dense1': _dense1,
        'fc1': _fc1,
        'fc1_drp': _fc1_drp,
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
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)
correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

# SAVER
save_step = 2
saver = tf.train.Saver(max_to_keep=2)
datadir = "./datadir_new"
if not os.path.exists(datadir):
    os.makedirs(datadir)

# actcode
# = 0, no read any data, epoch start from 1 to training_epochs. Save net para
# = 1, resume training, read data from file #data_index, epoch start from #epo_start to training_epochs. Save net para
# = 2, read data from file #data_index, no training, only inspect correction percentage
# = 3, read data from file #data_index, no training, inspect one by one

# para se1t
actcode = 2

epo_start = 1
training_epochs = 1000

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(init)

batch_size = 50
display_step = 1  # number of epoch print result
test_size = 100  # number used in test set

if actcode == 0:
    epo_start = 1

if actcode == 1 or actcode == 2 or actcode == 3 or actcode == 4:
    data_index = 552
    saver.restore(sess, datadir + "/myfirst.ckpt_" + str(data_index))

if actcode == 0 or actcode == 1:
    for epoch in range(epo_start, training_epochs + 1):
        time_start = time.time()
        # average cost of each batch
        avg_cost = 0.0
        # training_batch = zip(range(0, 1280, batch_size), range(batch_size, 1281, batch_size))
        training_batch = zip(range(0, len(X_Train), batch_size), range(batch_size, len(X_Train) + 1, batch_size))
        batch_num = 0
        for start, end in training_batch:
            batch_num = batch_num + 1
            sess.run(optimizer,
                     feed_dict={x: X_Train[start:end], y: Y_Train[start:end], p_keep_conv: 0.7, p_keep_hidden: 0.7})
            avg_cost += sess.run(cost, feed_dict={x: X_Train[start:end], y: Y_Train[start:end], p_keep_conv: 1.0,
                                                  p_keep_hidden: 1.0})

        # Training result output per epoch
        avg_cost = avg_cost / batch_num
        # acc_train = sess.run(accr, feed_dict={x: X_Train, y: Y_Train, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        acc_test = sess.run(accuracy, feed_dict={x: X_Test, y: Y_Test, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        time_end = time.time()
        print("epoch: %03d" % (epoch))
        print("Train time cost: %.6f" % (time_end - time_start))
        print("Train cost average: %.6f" % (avg_cost))
        # print("Test set accuracy: %.6f" % (acc_train))
        print("Test set accuracy: %.6f" % (acc_test))

        # Save Net
        if epoch % save_step == 0:
            saver.save(sess, datadir + "/myfirst.ckpt_" + str(epoch))

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
    print("Test set accuracy: %.6f" % (acc_test))


if actcode == 3:
    corr_test = sess.run(correct, feed_dict={x: x_test, y: y_test, p_keep_conv: 1.0, p_keep_hidden: 1.0})
    print(corr_test.shape)
    print(corr_test)
    for i in range(len(corr_test)):
        if not corr_test[i]:
            print("i in test set: %05d" % i)
            # print(tf.argmax(predy, 1).shape)
            # print(tf.argmax(teY[i], 1).shape)
            img = x_test[i].reshape(64, 64)
            for j in range(36):
                if y_test[i][j] == 1:
                    print(chr(j))
            # cv2.imshow("Image_COLOR", img)
            # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if actcode == 4:
    path_test = 'test/*'
    # for i in range(35):
    #     path_test = path_test+ ':' + 'test/' + str(i + 1) + '/*.png'
    coll_test = io.ImageCollection(path_test, load_func=resize)
    mat_test = io.concatenate_images(coll_test)
    # print(len(mat_test))
    # size (26*900, 128, 128)
    x_test = mat_test.reshape(-1, img_width, img_height, 1)  # size (26*900, 128, 128, 1)
    # conv1 = sess.run(output,feed_dict={x:x_test, p_keep_conv:1.0, p_keep_hidden:1.0})['conv1']
    # np.savetxt("/home/amax/PycharmProjects/Character Recognition/layers/c1",
    #            np.reshape((conv1_r), [-1, ]))
    # ref=np.reshape(np.loadtxt("/home/amax/PycharmProjects/Character Recognition/layers/conv1"),
    #            [-1,  64, 64, 8] )
    # pool1 = np.reshape(np.loadtxt("layers/pool1"), [1, 32, 32, 8])
    # print(sess.run(output,feed_dict={x:x_test, p_keep_conv:1.0, p_keep_hidden:1.0})['pool1'] == pool1)
    # print(conv1)

    # pred = np.argmax(pred1, 1)
    # pred=list(pred)
    # for k in  range(len(mat_test)):
    #     if pred[k]>9:
    #         pred[k] = chr(pred[k]+87)
    #
    # print(pred)
    # print(np.argmax(pred1, 1))
    t_vars = tf.trainable_variables()

    # print(np.array(sess.run(t_vars))[0].shape)

    for i in range(8):
        np.savetxt("/home/amax/PycharmProjects/character62/vars/var" + str(i), np.reshape(sess.run(t_vars)[i], [-1, ]))

    wc1 = np.reshape(np.loadtxt("vars/var0"), [3, 3, 1, 8])
    wc2 = np.reshape(np.loadtxt("vars/var1"), [3, 3, 8, 16])
    wd1 = np.reshape(np.loadtxt("vars/var2"), [8 * 8 * 16, 64])
    wd2 = np.reshape(np.loadtxt("vars/var3"), [64, n_output])
    bc1 = np.reshape(np.loadtxt("vars/var4"), [1, 1, 1, 8])
    bc2 = np.reshape(np.loadtxt("vars/var5"), [1, 1, 1, 16])
    bd1 = np.reshape(np.loadtxt("vars/var6"), [64])
    bd2 = np.reshape(np.loadtxt("vars/var7"), [n_output])

    print(sess.run(t_vars)[0] == wc1)
    print(sess.run(t_vars)[4] == bc1)

    #
    # biases_variable = reader.get_tensor('bc1')
    # weights_variable = reader.get_tensor('wc1')
    #
    # data = []
    # np.save(vars, data)
    # cv2.imshow("Image_COLOR", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # for i in range(0, len(corr_test)):
    #
    #     if ( corr_test[i] == False):
    #         # if tf.argmax(predy, 1) != tf.argmax(teY[i:i+1], 1):
    #         print("i in test set: %05d" % (i))
    #
    #         # print(tf.argmax(predy, 1).shape)
    #         # print(tf.argmax(teY[i], 1).shape)
    #         img = Image.new('L', (img_width, img_height), 255)  # 128,128 is length, width
    #         for ii in range(img_width):
    #             for jj in range(img_height):
    #                 # in MNIST, 0 is background/white, 1.0 is black
    #                 img.putpixel((jj, ii), 255 - int(X_Test[i][ii][jj] * 255.0))  # jj is length/colume; ii is width/row
    #         img.save('this' + str(i) + '.png')
    #         str0 = input("Enter number key 1: ")  # wait here until click 1
