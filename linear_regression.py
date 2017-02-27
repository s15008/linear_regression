# -*- encoding: utf-8 -*-

import tensorflow as tf
from matplotlib import pyplot

if __name__ == '__main__':
    """
    TensorFlowを試してみたよ
    Getting Startedで線形回帰のやり方があったよ
    やってみたよ
    training dataが点の座標になっているよ
    変更すると回帰式が変わってたのしいよ
    """
    # Model parameters
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)

    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)

    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # training data
    x_train = [1, 3, 4, 5]
    y_train = [-3, -4, -3, -5]

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})
        if i % 100 == 0:
            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
            print('WIP   W: %s b: %s loss: %s' % (curr_W, curr_b, curr_loss))

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print('W: %s b: %s loss: %s' % (curr_W, curr_b, curr_loss))


    # draw
    fig = pyplot.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    point = ax.plot(x_train, y_train, 'ro', label='data')
    x_min = curr_W * 0 + curr_b
    x_max = curr_W * 5 + curr_b
    line = ax.plot([0, 5], [x_min, x_max], label='line')

    pyplot.legend()
    pyplot.ylabel('Y-axis')
    pyplot.ylabel('X-axis')
    pyplot.xlim(-1, max(x_train)+1)
    pyplot.ylim(min(y_train)-1, 1)
    pyplot.show()
