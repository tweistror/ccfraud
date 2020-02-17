import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report, precision_recall_fscore_support

from advanced.oc_gan.utils import xavier_init, pull_away_loss, sample_shuffle_uspv, one_hot, sample_Z, draw_trend


def execute_oc_gan(dataset_string, x_ben, x_fraud, usv_train, test_fraud, verbosity=0):
    dim_input = x_ben.shape[1]

    # Set parameters
    if dataset_string == "paysim":
        usv_train = 2000
        test_fraud = 9000
        mb_size = 70
        D_dim = [dim_input, 30, 15, 2]
        G_dim = [15, 30, dim_input]
        Z_dim = G_dim[0]
    elif dataset_string == "ccfraud":
        mb_size = 70
        D_dim = [dim_input, 100, 50, 2]
        G_dim = [50, 100, dim_input]
        Z_dim = G_dim[0]
    elif dataset_string == "ieee":
        mb_size = 70

    # Set dimensions for discrimator, generator and


    # Define placeholders for labeled-data, unlabeled-data, noise-data and target-data.
    X_oc =  tf.compat.v1.placeholder(tf.float32, shape=[None, dim_input])
    Z = tf.compat.v1.placeholder(tf.float32, shape=[None, Z_dim])
    X_tar = tf.compat.v1.placeholder(tf.float32, shape=[None, dim_input])

    # Declare weights and biases of discriminator.
    D_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))
    D_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]]))

    D_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))
    D_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))

    D_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))
    D_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Declare weights and biases of generator.
    G_W1 = tf.Variable(xavier_init([G_dim[0], G_dim[1]]))
    G_b1 = tf.Variable(tf.zeros(shape=[G_dim[1]]))

    G_W2 = tf.Variable(xavier_init([G_dim[1], G_dim[2]]))
    G_b2 = tf.Variable(tf.zeros(shape=[G_dim[2]]))

    theta_G = [G_W1, G_W2, G_b1, G_b2]

    # Declare weights and biases of pre-train net for density estimation.
    T_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))
    T_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]]))

    T_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))
    T_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))

    T_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))
    T_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))

    theta_T = [T_W1, T_W2, T_W3, T_b1, T_b2, T_b3]

    def generator(z):
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_logit = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
        return G_logit

    def discriminator(x):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.softmax(D_logit)
        return D_prob, D_logit, D_h2

    # Pre-train net for density estimation.
    def discriminator_tar(x):
        T_h1 = tf.nn.relu(tf.matmul(x, T_W1) + T_b1)
        T_h2 = tf.nn.relu(tf.matmul(T_h1, T_W2) + T_b2)
        T_logit = tf.matmul(T_h2, T_W3) + T_b3
        T_prob = tf.nn.softmax(T_logit)
        return T_prob, T_logit, T_h2

    D_prob_real, D_logit_real, D_h2_real = discriminator(X_oc)

    G_sample = generator(Z)
    D_prob_gen, D_logit_gen, D_h2_gen = discriminator(G_sample)

    D_prob_tar, D_logit_tar, D_h2_tar = discriminator_tar(X_tar)
    D_prob_tar_gen, D_logit_tar_gen, D_h2_tar_gen = discriminator_tar(G_sample)

    # Discriminator loss
    y_real = tf.compat.v1.placeholder(tf.int32, shape=[None, D_dim[3]])
    y_gen = tf.compat.v1.placeholder(tf.int32, shape=[None, D_dim[3]])

    D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_real, labels=y_real))
    D_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_gen, labels=y_gen))

    ent_real_loss = -tf.reduce_mean(
        tf.reduce_sum(
            tf.multiply(D_prob_real, tf.math.log(D_prob_real)), 1
        )
    )

    ent_gen_loss = -tf.reduce_mean(
        tf.reduce_sum(
            tf.multiply(D_prob_gen, tf.math.log(D_prob_gen)), 1
        )
    )

    D_loss = D_loss_real + D_loss_gen + 1.85 * ent_real_loss

    # Generator loss
    pt_loss = pull_away_loss(D_h2_tar_gen)

    y_tar = tf.compat.v1.placeholder(tf.int32, shape=[None, D_dim[3]])
    T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_tar, labels=y_tar))
    tar_thrld = tf.divide(tf.reduce_max(D_prob_tar_gen[:, -1]) +
                          tf.reduce_min(D_prob_tar_gen[:, -1]), 2)

    indicator = tf.sign(
        tf.subtract(D_prob_tar_gen[:, -1],
                    tar_thrld))
    condition = tf.greater(tf.zeros_like(indicator), indicator)
    mask_tar = tf.where(condition, tf.zeros_like(indicator), indicator)
    G_ent_loss = tf.reduce_mean(tf.multiply(tf.math.log(D_prob_tar_gen[:, -1]), mask_tar))

    fm_loss = tf.reduce_mean(
        tf.sqrt(
            tf.reduce_sum(
                tf.square(D_logit_real - D_logit_gen), 1
            )
        )
    )

    G_loss = pt_loss + G_ent_loss + fm_loss

    D_solver = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=theta_D)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    T_solver = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(T_loss, var_list=theta_T)

    # Process data
    x_ben = sample_shuffle_uspv(x_ben)
    x_fraud = sample_shuffle_uspv(x_fraud)

    x_pre = x_ben[0:usv_train]
    y_pre = np.zeros(len(x_pre))
    y_pre = one_hot(y_pre, 2)

    x_train = x_pre

    y_real_mb = one_hot(np.zeros(mb_size), 2)
    y_fake_mb = one_hot(np.ones(mb_size), 2)

    x_test = x_pre[-test_fraud:].tolist() + x_fraud[-test_fraud:].tolist()
    x_test = np.array(x_test)
    y_test = np.zeros(len(x_test))
    y_test[test_fraud:] = 1

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Pre-training for target distribution
    _ = sess.run(T_solver,
                 feed_dict={
                     X_tar: x_pre,
                     y_tar: y_pre
                 })

    q = np.divide(len(x_train), mb_size)

    d_ben_pro, d_fake_pro, fm_loss_coll = list(), list(), list()
    f1_score = list()
    d_val_pro = list()
    n_round = 200

    for n_epoch in range(n_round):
        X_mb_oc = sample_shuffle_uspv(x_train)

        for n_batch in range(int(q)):
            _, D_loss_curr, ent_real_curr = sess.run([D_solver, D_loss, ent_real_loss],
                                                     feed_dict={
                                                         X_oc: X_mb_oc[n_batch * mb_size:(n_batch + 1) * mb_size],
                                                         Z: sample_Z(mb_size, Z_dim),
                                                         y_real: y_real_mb,
                                                         y_gen: y_fake_mb
                                                     })

            _, G_loss_curr, fm_loss_curr = sess.run([G_solver, G_loss, fm_loss],
                                                    feed_dict={Z: sample_Z(mb_size, Z_dim),
                                                               X_oc: X_mb_oc[n_batch * mb_size:(n_batch + 1) * mb_size],
                                                               })

        D_prob_real_, D_prob_gen_ = sess.run([D_prob_real, D_prob_gen],
                                             feed_dict={X_oc: x_train,
                                                        Z: sample_Z(len(x_train), Z_dim)})

        D_prob_fraud_ = sess.run(D_prob_real,
                                 feed_dict={X_oc: x_fraud[-test_fraud:]})

        d_ben_pro.append(np.mean(D_prob_real_[:, 0]))
        d_fake_pro.append(np.mean(D_prob_gen_[:, 0]))
        d_val_pro.append(np.mean(D_prob_fraud_[:, 0]))
        fm_loss_coll.append(fm_loss_curr)

        prob, _ = sess.run([D_prob_real, D_logit_real], feed_dict={X_oc: x_test})
        y_pred = np.argmax(prob, axis=1)
        conf_mat = classification_report(y_test, y_pred, target_names=['benign', 'fraud'], digits=4)
        f1_score.append(float(list(filter(None, conf_mat.strip().split(" ")))[12]))

    acc = np.sum(y_pred == y_test) / float(y_pred.shape[0])
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    if verbosity == 1:
        print(conf_mat)
        draw_trend(d_ben_pro, d_fake_pro, d_val_pro, fm_loss_coll, f1_score)

    return precision[0], recall[0], f1[0], acc, 'OC-GAN'

