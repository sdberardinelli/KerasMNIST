import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from vgg import getmodel
from keras import backend as K
from keras.datasets import mnist
from keras.utils.visualize_util import plot

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    K.set_session(sess)

    n_epochs = 100
    batch_size = 32
    img_channel = 1
    img_width, img_height = 28, 28
    n_classes = 10

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train[:, np.newaxis, :, :]
    Y_train = y_train[:, np.newaxis]

    model = getmodel(batch_size, img_channel, img_width, img_height, n_classes)

    plot(model, to_file='vgg.png')

    def get_next(_data, start, size):
        end = start+size
        if end >= _data.shape[0]:
            return False, _data[start:]
        else:
            return True, _data[start:end]

    x_i = 0
    x=list()
    y_a=list()
    y_l=list()
    f, axarr = plt.subplots(2, sharex=True)
    for i in range(n_epochs):

        count = 0

        while True:

            has_next, X = get_next(X_train, count, batch_size)
            has_next, Y = get_next(y_train, count, batch_size)
            count = count + X.shape[0]

            if not has_next:
                break

            startTime = int(round(time.time() * 1000))
            model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=False)

            loss, accuracy = model.evaluate(X, Y, verbose=False)
            x.append(x_i)
            y_a.append(accuracy)
            y_l.append(loss)
            x_i += 1

            if count % 1000 == 0:
                axarr[0].cla()
                axarr[0].plot(x, y_a)
                axarr[1].cla()
                axarr[1].plot(x, y_l)
                plt.pause(1e-10)

            endTime = int(round(time.time() * 1000))

            print('{:d} {:d} {:f}ms loss: {:f} accuracy {:f}'.format(i, count, endTime - startTime, loss, accuracy))

