
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
flags.DEFINE_float('min_score', 0.35, 'Don\'t stop before getting this score.')
flags.DEFINE_integer('max_steps', 10 ** 5, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')


class Classifier(object):

    def __init__(self):
        self.x_placeholder = tf.placeholder("float", shape=[None, IMAGE_PIXELS])
        self.y_placeholder = tf.placeholder("float", shape=[None, OUTPUT_CLASSES])
        self.keep_prob = tf.placeholder("float")
        self._model = self._build_model()
        self._session = tf.Session()
        self._saver = tf.train.Saver()

    def _build_model(self):
        # Convert input into 4D tensor
        x_image = tf.reshape(self.x_placeholder, [-1, 28, 28, 1])
        # Build first convolution layer
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        # Build second convolution layer
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        # Dense layer
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # Output layer
        W_out = self.weight_variable([1024, OUTPUT_CLASSES])
        b_out = self.bias_variable([OUTPUT_CLASSES])
        return tf.nn.softmax(tf.matmul(h_fc1_drop, W_out) + b_out)

    @staticmethod
    def loss(expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def train(self, train_data, test_data):
        cross_entropy = self.loss(self.y_placeholder, self._model)
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        self._session.run(init)
        best_score = 1
        not_improvement_count = 0
        for i in range(FLAGS.max_steps):
            images, labels = train_data.next_batch(FLAGS.batch_size)
            self._session.run(train_step, feed_dict={self.x_placeholder: images,
                                                     self.y_placeholder: labels,
                                                     self.keep_prob: 0.5})
            if i % 100 == 0:
                score = self.check_score(test_data)
                print("step %d, epoch: %d, score %g" % (i, train_data.epochs_completed, score))
                if score < FLAGS.min_score:
                    if score < best_score:
                        self.save_session()
                        best_score = score
                        not_improvement_count = 0
                    elif not_improvement_count < 5:
                        not_improvement_count += 1
                    else:
                        break
        self.save_session()

    def check_score(self, data):
        loss = self.loss(self.y_placeholder, self._model)
        score = self._session.run(loss, feed_dict={self.x_placeholder: data.images,
                                                   self.y_placeholder: data.labels,
                                                   self.keep_prob: 1.0})
        return score/data.num_examples

    def errors(self, data):
        self.restore_session()
        predictions = self._session.run(self._model, feed_dict={self.x_placeholder: data.images, self.keep_prob: 1.0})
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

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    datasets = read_data_sets(one_hot=True)
    clf = Classifier()
    clf.train(datasets.train, datasets.validation)
    test_accuracy = clf.check_score(datasets.test)
    print("Test score: %g, error: %g%%" % (test_accuracy, clf.errors(datasets.test)))
    print('Done.')
