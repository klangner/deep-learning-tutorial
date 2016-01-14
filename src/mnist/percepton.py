
import numpy as np
import tensorflow as tf
from input_data import read_data_sets


SESSION_FILE = '../../data/percepton.session'

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE
OUTPUT_CLASSES = 10

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('lambda1', 0, 'L1 Regularization coefficient.')
flags.DEFINE_float('lambda2', 1e-3, 'L2 Regularization coefficient.')
flags.DEFINE_float('min_score', 0.35, 'Don\'t stop before getting this score.')
flags.DEFINE_integer('max_steps', 10 ** 5, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 200, 'Batch size. Must divide evenly into the dataset sizes.')


class Classifier(object):

    def __init__(self):
        self.x_placeholder = tf.placeholder("float", shape=[None, IMAGE_PIXELS])
        self.y_placeholder = tf.placeholder("float", shape=[None, OUTPUT_CLASSES])
        self._model = self._build_model()
        self._session = tf.Session()
        self._saver = tf.train.Saver()

    def _build_model(self):
        # Hidden layer 1
        self._hw1 = self.weight_variable([IMAGE_PIXELS, FLAGS.hidden1], 1.0 / np.sqrt(float(IMAGE_PIXELS)))
        self._hb1 = self.bias_variable([FLAGS.hidden1])
        hidden1 = tf.nn.relu(tf.matmul(self.x_placeholder, self._hw1) + self._hb1)
        # Hidden layer 2
        self._hw2 = self.weight_variable([FLAGS.hidden1, FLAGS.hidden2], 1.0 / np.sqrt(float(FLAGS.hidden1)))
        self._hb2 = self.bias_variable([FLAGS.hidden2])
        hidden2 = tf.nn.relu(tf.matmul(hidden1, self._hw2) + self._hb2)
        # Output layer
        self._ow = self.weight_variable([FLAGS.hidden2, OUTPUT_CLASSES], 1.0 / np.sqrt(float(FLAGS.hidden2)))
        self._ob = self.bias_variable([OUTPUT_CLASSES])
        output = tf.nn.relu(tf.matmul(hidden2, self._ow) + self._ob)
        return tf.nn.softmax(output)

    @staticmethod
    def loss(expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def normalized_loss(self, expected, predicted):
        w1 = tf.reduce_sum(tf.abs(self._hw1)) + tf.reduce_sum(tf.abs(self._hw2)) + tf.reduce_sum(tf.abs(self._ow))
        l1 = FLAGS.lambda1 * w1/(FLAGS.hidden1 + FLAGS.hidden2 + OUTPUT_CLASSES)
        w2 = tf.reduce_sum(tf.pow(self._hw1, 2)) + tf.reduce_sum(tf.pow(self._hw2, 2)) + tf.reduce_sum(tf.pow(self._ow, 2))
        l2 = FLAGS.lambda2 * w2/(FLAGS.hidden1 + FLAGS.hidden2 + OUTPUT_CLASSES)
        return self.loss(expected, predicted) + l1 + l2

    def train(self, train_data, test_data):
        cross_entropy = self.normalized_loss(self.y_placeholder, self._model)
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        self._session.run(init)
        best_score = 1
        not_improvement_count = 0
        for i in range(FLAGS.max_steps):
            images, labels = train_data.next_batch(FLAGS.batch_size)
            self._session.run(train_step, feed_dict={self.x_placeholder: images, self.y_placeholder: labels})
            if i % 100 == 0:
                t1 = self.check_score(train_data)
                t2 = self.check_score(test_data)
                print("step %d, epoch: %d, train accuracy %g (test: %g)" % (i, train_data.epochs_completed, t1, t2))
                if t2 < FLAGS.min_score:
                    if t2 < best_score:
                        self.save_session()
                        best_score = t2
                        not_improvement_count = 0
                    elif not_improvement_count < 5:
                        not_improvement_count += 1
                    else:
                        break
        self.save_session()

    def check_score(self, data):
        loss = self.loss(self.y_placeholder, self._model)
        score = self._session.run(loss, feed_dict={self.x_placeholder: data.images, self.y_placeholder: data.labels})
        return score/data.num_examples

    def errors(self, data):
        self.restore_session()
        predictions = self._session.run(self._model, feed_dict={self.x_placeholder: data.images})
        e = np.argmax(data.labels, axis=1)
        p = np.argmax(predictions, axis=1)
        return (np.not_equal(p, e).sum()*100.0)/len(p)

    def save_session(self):
        self._saver.save(self._session, SESSION_FILE)

    def restore_session(self):
        init = tf.initialize_all_variables()
        self._session.run(init)
        self._saver.restore(self._session, SESSION_FILE)

    @staticmethod
    def weight_variable(shape, std=0.1):
        initial = tf.truncated_normal(shape, stddev=std)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape, std=0.1):
        initial = tf.constant(std, shape=shape)
        return tf.Variable(initial)


# Logistic regression error = &.53%
if __name__ == "__main__":
    datasets = read_data_sets(one_hot=True)
    clf = Classifier()
    clf.train(datasets.train, datasets.validation)
    test_accuracy = clf.check_score(datasets.test)
    print("Test error: %g%%" % clf.errors(datasets.test))
    print('Done.')
