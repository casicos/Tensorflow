import tensorflow as tf
import numpy as np

xy = np.loadtxt('dataSetModified.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 1:-1]
y_data = xy[:, -1:]

np.random.shuffle(x_data)

X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis + 1e-6) + (1 - Y) * tf.log(1 - hypothesis + 1e-6))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100000 + 1):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 2000 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})

    print("\nHypothesis: ", hypothesis, "\nAccuracy: ", a)
    for idx in range(len(x_data)):
        print(idx, 'predicted: ', c[idx], '-> answer: ', y_data[idx])