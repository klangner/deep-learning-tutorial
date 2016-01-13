
import numpy as np
import tensorflow as tf
from input_data import read_data_sets


IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE
OUTPUT_CLASSES = 10

LEARNING_RATE = 1e-4
STEPS = 10 ** 4


class LogisticRegression(object):

    def __init__(self):
        self.x_placeholder = tf.placeholder("float", shape=[None, IMAGE_PIXELS])
        self.y_placeholder = tf.placeholder("float", shape=[None, OUTPUT_CLASSES])
        self._model = self._build_model()
        self._session = tf.Session()

    def _build_model(self):
        weights = self.weight_variable([IMAGE_PIXELS, OUTPUT_CLASSES])
        biases = self.bias_variable([OUTPUT_CLASSES])
        return tf.nn.softmax(tf.matmul(self.x_placeholder, weights) + biases)

    @staticmethod
    def loss(expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def train(self, train_data, test_data):
        cross_entropy = self.loss(self.y_placeholder, self._model)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        self._session.run(init)
        for i in range(STEPS):
            images, labels = train_data.next_batch(200)
            self._session.run(train_step, feed_dict={self.x_placeholder: images, self.y_placeholder: labels})
            if i % 100 == 0:
                train_accuracy = self.check_score(train_data)
                test_accuracy = self.check_score(test_data)
                print("step %d, train accuracy %g (test: %g)" % (i, train_accuracy, test_accuracy))

    def check_score(self, data):
        loss = self.loss(self.y_placeholder, self._model)
        score = self._session.run(loss, feed_dict={self.x_placeholder: data.images, self.y_placeholder: data.labels})
        return score/data.num_examples

    def errors(self, data):
        predictions = self._session.run(self._model, feed_dict={self.x_placeholder: data.images})
        e = np.argmax(data.labels, axis=1)
        p = np.argmax(predictions, axis=1)
        return (np.equal(p, e).sum()*100.0)/len(p)

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


if __name__ == "__main__":
    datasets = read_data_sets(one_hot=True)
    lr = LogisticRegression()
    lr.train(datasets.train, datasets.validation)
    test_accuracy = lr.check_score(datasets.test)
    print("Score: %g%%" % lr.errors(datasets.test))
    print('Done.')
