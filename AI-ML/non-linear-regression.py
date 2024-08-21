#non-linear regression implementation
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]


noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise


x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])


Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + biases_L1

 # Activation function tanh
L1 = tf.nn.tanh(Wx_plus_b_L1)

 # Define the neural network output layer
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + biases_L2

prediction = tf.nn.tanh(Wx_plus_b_L2)

 # secondary cost function
loss = tf.reduce_mean(tf.square(y - prediction))

 # Training using gradient descent
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for step in range(2000):
                 sess.run(train_step, feed_dict={x: x_data, y: y_data}) #pass in sample value

         #Get predicted value
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

         # 
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()