# Taken from https://link.springer.com/content/pdf/10.1007%2F978-1-4842-5177-5.pdf and
# https://github.com/aaxwaz/Fraud-detection-using-deep-learning with some modifications

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve, roc_auc_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split

from advanced_methods.RBM import utils

tf.disable_v2_behavior()


class RBM(object):
    """ Restricted Boltzmann Machine implementation using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, dataset_string, seed, verbosity=0, plot_training_loss=False):
        self.dataset_string = dataset_string
        tf.set_random_seed(seed)
        self.seed = seed

        self.rec_error = None
        self.reconstruction = None
        self.label = 'RBM'
        self.cm = None

        self.visible_unit_type = 'bin'
        self.gibbs_sampling_steps = None
        self.learning_rate = None
        self.momentum = None
        self.l2 = None
        self.batch_size = None
        self.num_epochs = None
        self.stddev = None
        self.train_test_split = None

        self.verbosity = 1 if verbosity == 2 else 0

        self.plot_training_loss = plot_training_loss

        self.W = None
        self.bh_ = None
        self.bv_ = None
        self.dw = None
        self.dbh_ = None
        self.dbv_ = None

        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None

        self.encode = None
        self.reconstruct = None

        self.loss_function = None
        self.batch_cost = None
        self.batch_free_energy = None

        self.training_losses = []

        self.input_data = None
        self.hrand = None
        self.validation_size = None

        self.tf_session = None

        self.num_visible = None
        self.num_hidden = None

    def set_parameters(self, num_visible, parameters):
        self.num_visible = num_visible
        self.num_hidden = parameters['num_hidden']

        self.visible_unit_type = parameters['visible_unit_type']
        self.gibbs_sampling_steps = parameters['gibbs_sampling_steps']
        self.learning_rate = parameters['learning_rate']
        self.momentum = parameters['momentum']
        self.l2 = parameters['l2']
        self.batch_size = parameters['batch_size']
        self.num_epochs = parameters['epochs']
        self.stddev = parameters['stddev']
        self.train_test_split = parameters['train_test_split']

    def execute(self, x_train, x_test, y_test):

        """ Execute the model with given training and test data.
        :param x_train: training set
        :param x_test: testing set
        :param y_test: testing labels
        :return: self
        """

        x_train_split, x_valid_split = train_test_split(x_train, test_size=self.train_test_split,
                                                        random_state=self.seed)

        tf.reset_default_graph()

        self._build_model()

        with tf.Session() as self.tf_session:
            self._initialize_tf_utilities_and_ops()
            self._train_model(x_train_split, x_valid_split)

            if self.plot_training_loss:
                plt.plot(self.training_losses)
                plt.title("Training batch losses v.s. iteractions")
                plt.xlabel("Num of training iteractions")
                plt.ylabel("Reconstruction error")
                plt.show()

            return self.predict(x_train, x_test, y_test)

    def _initialize_tf_utilities_and_ops(self):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        """

        init_op = tf.global_variables_initializer()

        self.tf_session.run(init_op)

    def _train_model(self, train_set, validation_set):

        """ Train the model.
        :param train_set: training set
        :param validation_set: validation set. optional, default None
        :return: self
        """

        for i in range(self.num_epochs):
            self._run_train_step(train_set)

            if validation_set is not None:
                self._run_validation_error(i, validation_set)

    def _run_train_step(self, train_set):

        """ Run a training step. A training step is made by randomly shuffling the training set,
        divide into batches and run the variable update nodes for each batch. If self.plot_training_loss
        is true, will record training loss after each batch.
        :param train_set: training set
        :return: self
        """

        np.random.seed(self.seed)
        np.random.shuffle(train_set)

        batches = [_ for _ in utils.gen_batches(train_set, self.batch_size)]
        updates = [self.w_upd8, self.bh_upd8, self.bv_upd8]

        for batch in batches:
            if self.plot_training_loss:
                _, loss = self.tf_session.run([updates, self.loss_function], feed_dict=self._create_feed_dict(batch))
                self.training_losses.append(loss)
            else:
                self.tf_session.run(updates, feed_dict=self._create_feed_dict(batch))

    def _run_validation_error(self, epoch, validation_set):

        """ Run the error computation on the validation set and print it out for each epoch.
        :param epoch: current epoch
        :param validation_set: validation data
        :return: self
        """

        loss = self.tf_session.run(self.loss_function,
                                   feed_dict=self._create_feed_dict(validation_set))

        if self.verbosity == 1:
            print("Validation cost at step %s: %s" % (epoch, loss))

    def _create_feed_dict(self, data):

        """ Create the dictionary of data to feed to TensorFlow's session during training.
        :param data: training/validation set batch
        :return: dictionary(self.input_data: data, self.hrand: random_uniform)
        """

        np.random.seed(self.seed)

        return {
            self.input_data: data,
            self.hrand: np.random.rand(data.shape[0], self.num_hidden),
        }

    def _build_model(self):

        """ Build the Restricted Boltzmann Machine model in TensorFlow.
        :return: self
        """

        self.input_data, self.hrand = self._create_placeholders()
        self.W, self.bh_, self.bv_, self.dw, self.dbh_, self.dbv_ = self._create_variables()

        hprobs0, hstates0, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(self.input_data)
        positive = self.compute_positive_association(self.input_data, hprobs0, hstates0)

        nn_input = vprobs

        for step in range(self.gibbs_sampling_steps - 1):
            hprobs, hstates, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(nn_input)
            nn_input = vprobs

        self.reconstruct = vprobs

        negative = tf.matmul(tf.transpose(vprobs), hprobs1)

        self.encode = hprobs1  # encoded data, used by the transform method

        dw = positive - negative
        self.dw = self.momentum * self.dw + (1 - self.momentum) * dw
        self.w_upd8 = self.W.assign_add(self.learning_rate * self.dw - self.learning_rate * self.l2 * self.W)

        dbh_ = tf.reduce_mean(hprobs0 - hprobs1, 0)
        self.dbh_ = self.momentum * self.dbh_ + self.learning_rate * dbh_
        self.bh_upd8 = self.bh_.assign_add(self.dbh_)

        dbv_ = tf.reduce_mean(self.input_data - vprobs, 0)
        self.dbv_ = self.momentum * self.dbv_ + self.learning_rate * dbv_
        self.bv_upd8 = self.bv_.assign_add(self.dbv_)

        self.loss_function = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - vprobs)))

        self.batch_cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - vprobs), 1))

        self._create_free_energy_for_batch()

    def _create_free_energy_for_batch(self):

        """ Create free energy ops to batch input data
        :return: self
        """

        if self.visible_unit_type == 'bin':
            self._create_free_energy_for_bin()
        elif self.visible_unit_type == 'gauss':
            self._create_free_energy_for_gauss()
        else:
            self.batch_free_energy = None

    def _create_free_energy_for_bin(self):

        """ Create free energy for mdoel with Bin visible layer
        :return: self
        """

        self.batch_free_energy = - (tf.matmul(self.input_data, tf.reshape(self.bv_, [-1, 1])) +
                                    tf.reshape(
                                        tf.reduce_sum(tf.log(tf.exp(tf.matmul(self.input_data, self.W) + self.bh_) + 1),
                                                      1), [-1, 1]))

    def _create_free_energy_for_gauss(self):

        """ Create free energy for model with Gauss visible layer
        :return: self
        """

        self.batch_free_energy = - (tf.matmul(self.input_data, tf.reshape(self.bv_, [-1, 1])) -
                                    tf.reshape(tf.reduce_sum(0.5 * self.input_data * self.input_data, 1), [-1, 1]) +
                                    tf.reshape(
                                        tf.reduce_sum(tf.log(tf.exp(tf.matmul(self.input_data, self.W) + self.bh_) + 1),
                                                      1), [-1, 1]))

    def _create_placeholders(self):

        """ Create the TensorFlow placeholders for the model.
        :return: tuple(input(shape(None, num_visible)),
                       hrand(shape(None, num_hidden)))
        """

        x = tf.placeholder('float', [None, self.num_visible], name='x-input')
        hrand = tf.placeholder('float', [None, self.num_hidden], name='hrand')

        return x, hrand

    def _create_variables(self):

        """ Create the TensorFlow variables for the model.
        :return: tuple(weights(shape(num_visible, num_hidden),
                       hidden bias(shape(num_hidden)),
                       visible bias(shape(num_visible)))
        """

        tf.set_random_seed(self.seed)
        W = tf.Variable(tf.random_normal((self.num_visible, self.num_hidden), mean=0.0, stddev=0.01), name='weights')
        dw = tf.Variable(tf.zeros([self.num_visible, self.num_hidden]), name='derivative-weights')

        bh_ = tf.Variable(tf.zeros([self.num_hidden]), name='hidden-bias')
        dbh_ = tf.Variable(tf.zeros([self.num_hidden]), name='derivative-hidden-bias')

        bv_ = tf.Variable(tf.zeros([self.num_visible]), name='visible-bias')
        dbv_ = tf.Variable(tf.zeros([self.num_visible]), name='derivative-visible-bias')

        return W, bh_, bv_, dw, dbh_, dbv_

    def gibbs_sampling_step(self, visible):

        """ Performs one step of gibbs sampling.
        :param visible: activations of the visible units
        :return: tuple(hidden probs, hidden states, visible probs,
                       new hidden probs, new hidden states)
        """

        hprobs, hstates = self.sample_hidden_from_visible(visible)
        vprobs = self.sample_visible_from_hidden(hprobs)
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def sample_hidden_from_visible(self, visible):

        """ Sample the hidden units from the visible units.
        This is the Positive phase of the Contrastive Divergence algorithm.
        :param visible: activations of the visible units
        :return: tuple(hidden probabilities, hidden binary states)
        """

        hprobs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.bh_)
        hstates = utils.sample_prob(hprobs, self.hrand)

        return hprobs, hstates

    def sample_visible_from_hidden(self, hidden):

        """ Sample the visible units from the hidden units.
        This is the Negative phase of the Contrastive Divergence algorithm.
        :param hidden: activations of the hidden units
        :return: visible probabilities
        """

        visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.bv_

        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal((1, self.num_visible), mean=visible_activation, stddev=self.stddev)
        else:
            vprobs = None

        return vprobs

    def compute_positive_association(self, visible, hidden_probs, hidden_states):

        """ Compute positive associations between visible and hidden units.
        :param visible: visible units
        :param hidden_probs: hidden units probabilities
        :param hidden_states: hidden units states
        :return: positive association = dot(visible.T, hidden)
        """

        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None

        return positive

    def getReconstructError(self, data):

        """ return Reconstruction Error (loss) from data in batch.
        :param data: input data of shape num_samples x visible_size
        :return: Reconstruction cost for each sample in the batch
        """

        batch_loss = self.tf_session.run(self.batch_cost, feed_dict=self._create_feed_dict(data))
        return batch_loss

    def getFreeEnergy(self, data):

        """ return Free Energy from data.
        :param data: input data of shape num_samples x visible_size
        :return: Free Energy for each sample: p(x)
        """

        with tf.Session() as self.tf_session:
            batch_FE = self.tf_session.run(self.batch_free_energy,
                                           feed_dict=self._create_feed_dict(data))

            return batch_FE

    def getReconstruction(self, data):

        batch_reconstruct = self.tf_session.run(self.reconstruct,
                                                feed_dict=self._create_feed_dict(data))

        return batch_reconstruct

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        with tf.Session() as self.tf_session:
            return {
                'W': self.W.eval(),
                'bh_': self.bh_.eval(),
                'bv_': self.bv_.eval()
            }

    def predict(self, x_train, x_test, y_test):
        # Threshold calculation
        train_reconstruction_errors = self.getReconstructError(x_train)
        threshold = np.quantile(train_reconstruction_errors, 0.9)

        rec_error = self.getReconstructError(x_test)
        self.rec_error = rec_error

        # TODO: Move this into `plot_reconstructed_images`-method (“RuntimeError: Attempted to use a closed Session”)
        self.reconstruction = self.getReconstruction(x_test)

        y_pred = [1 if val > threshold else 0 for val in rec_error]
        acc_score = accuracy_score(y_test, y_pred)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
        self.cm = confusion_matrix(y_test, y_pred)

        precision_pts, recall_pts, _ = precision_recall_curve(y_test, rec_error)
        pr_auc = metrics.auc(recall_pts, precision_pts)
        roc_auc = roc_auc_score(y_test, rec_error)

        results = {
            'prec_list': [precision[1]],
            'reca_list': [recall[1]],
            'f1_list': [fscore[1]],
            'acc_list': [acc_score],
            'pr_auc_list': [pr_auc],
            'roc_auc_list': [roc_auc],
            'method_list': [self.label],
        }

        return results

    def build_plots(self, y_test, image_creator):
        image_creator.add_curves(y_test, self.rec_error, self.label)

    def plot_reconstructed_images(self, x_test, image_creator):
        reconstructed_x_test = self.reconstruction
        image_creator.add_image_plots(x_test, reconstructed_x_test, self.label, self.dataset_string, 10)

    def plot_conf_matrix(self, image_creator):
        image_creator.plot_conf_matrix(self.cm, self.label)
