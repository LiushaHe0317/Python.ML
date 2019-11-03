import numpy as np
import tensorflow as tf

## --- session
D = 10
N = 8
T = 30

t_1 = np.random.randn(D, N, T)
t_2 = np.random.randn(D, N, T)

mul = tf.add(t_1, t_2)

sess = tf.Session()
t_3 = sess.run(mul)

print('{} + {} = '.format(t_1[1][1][1], t_2[1][1][1]))
print(t_3[1][1][1])
print(t_1[1][1][1] + t_2[1][1][1])

## --- constant and variable
x = tf.constant(35, name='x')
y = tf.Variable(x+5, name='y')

model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    print(sess.run(y))

## --- placeholder
x = tf.placeholder(tf.int32, shape=[D,N,T])
y = tf.placeholder(tf.int32, shape = [D,N,T])

sum_x = tf.reduce_sum(x)
prod_y = tf.reduce_prod(y)
final_mean = tf.reduce_mean([sum_x, prod_y])

sess = tf.Session()
print(sess.run(sum_x, feed_dict={x:t_1}))
print(sess.run(prod_y, feed_dict={y:t_2}))