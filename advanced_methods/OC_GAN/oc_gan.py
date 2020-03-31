# Taken from https://github.com/PanpanZheng/OCAN with some modifications

import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

from advanced_methods.OC_GAN.autoencoder import Dense_Autoencoder
from advanced_methods.OC_GAN.utils import xavier_init, pull_away_loss, sample_shuffle_uspv, one_hot, sample_Z, draw_trend, \
    preprocess_minus_1_and_pos_1

tf.compat.v1.disable_eager_execution()


def execute_oc_gan(dataset_string, x_usv_train, x_test_benign, x_test_fraud, n_test_benign, autoencoding=False,
                   verbosity=0):
    # Set parameters
    dim_input = x_usv_train.shape[1]
    if dataset_string == "paysim" or dataset_string == "paysim_custom":
        mb_size = 70
        d_dim = [dim_input, 30, 15, 2]
        g_dim = [15, 30, dim_input]
        z_dim = g_dim[0]
        hid_dim = [30]
    elif dataset_string == "ccfraud":
        mb_size = 70
        d_dim = [dim_input, 100, 50, 2]
        g_dim = [50, 100, dim_input]
        z_dim = g_dim[0]
        hid_dim = [100]
    elif dataset_string == "ieee":
        mb_size = 70
        d_dim = [dim_input, 256, 64, 2]
        g_dim = [64, 256, dim_input]
        z_dim = g_dim[0]
        hid_dim = [400]

    if autoencoding is True:
        x_ben = np.concatenate((x_usv_train, x_test_benign))
        x_fraud = x_test_fraud
        dense_ae = Dense_Autoencoder(dim_input, hid_dim, verbosity)
        dense_ae.compile()
        dense_ae.fit(x_usv_train)
        dense_ae.get_hidden_layer()
        ben_hid_repre, van_hid_repre = list(map(lambda x: dense_ae.get_hidden_representation(x), [
            x_ben, x_fraud]))
        ben_hid_repre, van_hid_repre = list(map(lambda x: preprocess_minus_1_and_pos_1(x),
                                                [ben_hid_repre, van_hid_repre]))

        # Set additional parameters for autoencoding
        if dataset_string == "paysim" or dataset_string == "paysim_custom":
            dim_input = 15
            d_dim = [dim_input, 30, 15, 2]
            g_dim = [15, 30, dim_input]
            z_dim = g_dim[0]
        elif dataset_string == "ccfraud":
            dim_input = 50
            d_dim = [dim_input, 100, 50, 2]
            g_dim = [50, 100, dim_input]
            z_dim = g_dim[0]
        elif dataset_string == "ieee":
            dim_input = 200
            d_dim = [dim_input, 256, 64, 2]
            g_dim = [64, 256, dim_input]
            z_dim = g_dim[0]

    # Define placeholders for labeled-data, unlabeled-data, noise-data and target-data.
    x_oc = tf.compat.v1.placeholder(tf.float32, shape=[None, dim_input])
    z = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim])
    x_tar = tf.compat.v1.placeholder(tf.float32, shape=[None, dim_input])

    # Declare weights and biases of discriminator.
    d_w1 = tf.Variable(xavier_init([d_dim[0], d_dim[1]]))
    d_b1 = tf.Variable(tf.zeros(shape=[d_dim[1]]))

    d_w2 = tf.Variable(xavier_init([d_dim[1], d_dim[2]]))
    d_b2 = tf.Variable(tf.zeros(shape=[d_dim[2]]))

    d_w3 = tf.Variable(xavier_init([d_dim[2], d_dim[3]]))
    d_b3 = tf.Variable(tf.zeros(shape=[d_dim[3]]))

    theta_d = [d_w1, d_w2, d_w3, d_b1, d_b2, d_b3]

    # Declare weights and biases of generator.
    g_w1 = tf.Variable(xavier_init([g_dim[0], g_dim[1]]))
    g_b1 = tf.Variable(tf.zeros(shape=[g_dim[1]]))

    g_w2 = tf.Variable(xavier_init([g_dim[1], g_dim[2]]))
    g_b2 = tf.Variable(tf.zeros(shape=[g_dim[2]]))

    theta_g = [g_w1, g_w2, g_b1, g_b2]

    # Declare weights and biases of pre-train net for density estimation.
    t_w1 = tf.Variable(xavier_init([d_dim[0], d_dim[1]]))
    t_b1 = tf.Variable(tf.zeros(shape=[d_dim[1]]))

    t_w2 = tf.Variable(xavier_init([d_dim[1], d_dim[2]]))
    t_b2 = tf.Variable(tf.zeros(shape=[d_dim[2]]))

    t_w3 = tf.Variable(xavier_init([d_dim[2], d_dim[3]]))
    t_b3 = tf.Variable(tf.zeros(shape=[d_dim[3]]))

    theta_t = [t_w1, t_w2, t_w3, t_b1, t_b2, t_b3]

    def generator(z):
        g_h1 = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)
        g_logit = tf.nn.tanh(tf.matmul(g_h1, g_w2) + g_b2)
        return g_logit

    def discriminator(x):
        d_h1 = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
        d_h2 = tf.nn.relu(tf.matmul(d_h1, d_w2) + d_b2)
        d_logit = tf.matmul(d_h2, d_w3) + d_b3
        d_prob = tf.nn.softmax(d_logit)
        return d_prob, d_logit, d_h2

    # Pre-train net for density estimation.
    def discriminator_tar(x):
        t_h1 = tf.nn.relu(tf.matmul(x, t_w1) + t_b1)
        t_h2 = tf.nn.relu(tf.matmul(t_h1, t_w2) + t_b2)
        t_logit = tf.matmul(t_h2, t_w3) + t_b3
        t_prob = tf.nn.softmax(t_logit)
        return t_prob, t_logit, t_h2

    d_prob_real, d_logit_real, d_h2_real = discriminator(x_oc)

    g_sample = generator(z)
    d_prob_gen, d_logit_gen, d_h2_gen = discriminator(g_sample)

    d_prob_tar, d_logit_tar, d_h2_tar = discriminator_tar(x_tar)
    d_prob_tar_gen, d_logit_tar_gen, d_h2_tar_gen = discriminator_tar(g_sample)

    # Discriminator loss
    y_real = tf.compat.v1.placeholder(tf.int32, shape=[None, d_dim[3]])
    y_gen = tf.compat.v1.placeholder(tf.int32, shape=[None, d_dim[3]])

    d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logit_real, labels=y_real))
    d_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logit_gen, labels=y_gen))

    ent_real_loss = -tf.reduce_mean(
        tf.reduce_sum(
            tf.multiply(d_prob_real, tf.math.log(d_prob_real)), 1
        )
    )

    ent_gen_loss = -tf.reduce_mean(
        tf.reduce_sum(
            tf.multiply(d_prob_gen, tf.math.log(d_prob_gen)), 1
        )
    )

    d_loss = d_loss_real + d_loss_gen + 1.85 * ent_real_loss

    # Generator loss
    pt_loss = pull_away_loss(d_h2_tar_gen)

    y_tar = tf.compat.v1.placeholder(tf.int32, shape=[None, d_dim[3]])
    t_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logit_tar, labels=y_tar))
    tar_thrld = tf.divide(tf.reduce_max(d_prob_tar_gen[:, -1]) +
                          tf.reduce_min(d_prob_tar_gen[:, -1]), 2)

    indicator = tf.sign(
        tf.subtract(d_prob_tar_gen[:, -1],
                    tar_thrld))
    condition = tf.greater(tf.zeros_like(indicator), indicator)
    mask_tar = tf.where(condition, tf.zeros_like(indicator), indicator)
    g_ent_loss = tf.reduce_mean(tf.multiply(tf.math.log(d_prob_tar_gen[:, -1]), mask_tar))

    fm_loss = tf.reduce_mean(
        tf.sqrt(
            tf.reduce_sum(
                tf.square(d_logit_real - d_logit_gen), 1
            )
        )
    )

    g_loss = pt_loss + g_ent_loss + fm_loss

    d_solver = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(d_loss, var_list=theta_d)
    g_solver = tf.compat.v1.train.AdamOptimizer().minimize(g_loss, var_list=theta_g)
    t_solver = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(t_loss, var_list=theta_t)

    # Process data
    x_pre = x_usv_train if autoencoding is False else ben_hid_repre[:-n_test_benign]
    x_fraud = x_test_fraud if autoencoding is False else van_hid_repre

    y_pre = np.zeros(len(x_pre))
    y_pre = one_hot(y_pre, 2)

    x_train = x_pre

    y_real_mb = one_hot(np.zeros(mb_size), 2)
    y_fake_mb = one_hot(np.ones(mb_size), 2)

    x_test = np.concatenate((x_test_benign, x_fraud)) if autoencoding is False else \
        np.concatenate((ben_hid_repre[-n_test_benign:], x_fraud))

    y_test = np.zeros(len(x_test))
    y_test[n_test_benign:] = 1

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Pre-training for target distribution
    _ = sess.run(t_solver,
                 feed_dict={
                     x_tar: x_pre,
                     y_tar: y_pre
                 })

    q = np.divide(len(x_train), mb_size)

    d_ben_pro, d_fake_pro, fm_loss_coll = list(), list(), list()
    f1_score = list()
    d_val_pro = list()
    n_round = 200

    for n_epoch in range(n_round):
        x_mb_oc = sample_shuffle_uspv(x_train)

        for n_batch in range(int(q)):
            _, d_loss_curr, ent_real_curr = sess.run([d_solver, d_loss, ent_real_loss],
                                                     feed_dict={
                                                         x_oc: x_mb_oc[n_batch * mb_size:(n_batch + 1) * mb_size],
                                                         z: sample_Z(mb_size, z_dim),
                                                         y_real: y_real_mb,
                                                         y_gen: y_fake_mb
                                                     })

            _, g_loss_curr, fm_loss_curr = sess.run([g_solver, g_loss, fm_loss],
                                                    feed_dict={z: sample_Z(mb_size, z_dim),
                                                               x_oc: x_mb_oc[n_batch * mb_size:(n_batch + 1) * mb_size],
                                                               })

        d_prob_real_, d_prob_gen_ = sess.run([d_prob_real, d_prob_gen],
                                             feed_dict={x_oc: x_train,
                                                        z: sample_Z(len(x_train), z_dim)})

        d_prob_fraud_ = sess.run(d_prob_real,
                                 feed_dict={x_oc: x_test[-n_test_benign:]})

        d_ben_pro.append(np.mean(d_prob_real_[:, 0]))
        d_fake_pro.append(np.mean(d_prob_gen_[:, 0]))
        d_val_pro.append(np.mean(d_prob_fraud_[:, 0]))
        fm_loss_coll.append(fm_loss_curr)

        prob, _ = sess.run([d_prob_real, d_logit_real], feed_dict={x_oc: x_test})
        y_pred = np.argmax(prob, axis=1)
        conf_mat = classification_report(y_test, y_pred, target_names=['benign', 'fraud'], digits=4, zero_division=0)
        f1_score.append(float(list(filter(None, conf_mat.strip().split(" ")))[12]))

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    # print(conf_mat)
    # draw_trend(d_ben_pro, d_fake_pro, d_val_pro, fm_loss_coll, f1_score)

    return precision[1], recall[1], f1[1], acc, f'OC-GAN{" with AE" if autoencoding is True else ""}'

