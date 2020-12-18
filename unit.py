import tensorflow as tf  # 导入tensorflow库并定义简写别名
import numpy as np  # 导入numpy库，因为深度学习计算是基于矩阵的运算

# 定义变量
weight = tf.Variable(100)
bias = tf.Variable(5)
# 用占位符定义输入
x_input = tf.placeholder(tf.int32, [None])
# 定义计算流
# 有些基本运算符可以简化写法：prediction = weight * x_input + bias
prediction = tf.add(tf.multiply(weight, x_input), bias)
# 看看直接打印prediction会获得什么
# 获得的是一个Tensor对象：Tensor("Add:0", shape=(?,), dtype=int32)
print(prediction)

# 准备输入数据
x_data = np.array([3])

# 初始化变量，并定义一个运算session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 用feed_dict喂入输入数据，通过sess.run获得运算结果
result = sess.run(prediction, feed_dict={x_input: x_data})
# 打印结果，获得[305]
print(result)


