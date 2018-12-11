from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate, subtract, add, dot, Lambda, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras import backend as K
from scipy.sparse import csr_matrix
from keras.layers.convolutional import UpSampling2D, Conv2D

from tensorflow import boolean_mask

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from gcn.gcn_layer import GCNLayer

from utils import *

import os

MAX_DEPTH = 3

class ACGAN():
    def __init__(self, feature_dim, num_nodes, num_classes, latent_dim=32, dropout_rate=0., mlp_units=(64, 128, 256), clip_value=0.01):
        # Input shape
        self.latent_dim = latent_dim

        self.feature_dim = feature_dim
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.mlp_units = mlp_units
        self.clip_value = clip_value

        optimizer = RMSprop(0.00005)
        losses = [self.wasserstein_loss, self.wasserstein_loss]

        # Build and compile the discriminator
        self.discriminator_unlabeled = self.build_discriminator_unlabeled()
        self.discriminator_unlabeled.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))

        gen_graph = self.generator([label, noise])

        # For the combined model we will only train the generator
        self.discriminator_unlabeled.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image

        valid,target_label = self.discriminator_unlabeled(gen_graph)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([label, noise], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

        # load saved model
        if(os.listdir("saved_model")):
            print("Loading weights from saved_model/")
            self.generator.load_weights("saved_model/generator_weights.hdf5")
            self.discriminator_unlabeled.load_weights("saved_model/discriminator_unlabeled_weights.hdf5")
            print("Done.")

    def build_generator(self):
        """
        directly concat label info to feature info
        :return:
        """
        label = Input(shape=(1,))
        label_embedding = Embedding(self.num_classes, self.latent_dim, input_length=1)(label)
        label_embedding = Reshape(target_shape=(self.latent_dim,))(label_embedding)
        noise = Input(shape=(self.latent_dim,))
        label_in = multiply([label_embedding, noise])

        # generative model.
        #
        # MLP
        mlp = label_in
        for u in self.mlp_units:
            mlp = Dense(u)(mlp)
            mlp = BatchNormalization()(mlp)
            mlp = Activation("relu")(mlp)

        # generate adj
        features = Reshape((self.num_nodes, self.feature_dim))(Dense(self.num_nodes*self.feature_dim,)(mlp))
        features = Activation("tanh")(features)

        adj = Reshape((self.num_nodes, self.num_nodes))(Dense(self.num_nodes*self.num_nodes)(mlp))
        adj = Lambda(lambda x: (x+tf.matrix_transpose(x))/2)(adj)
        adj = Activation("sigmoid")(adj)

        # Compile model
        model = Model(inputs=[label, noise], outputs=[features, adj])
        model.summary()

        return model

    def build_discriminator_unlabeled(self):

        adj = Input(shape=(self.num_nodes, self.num_nodes,))
        features = Input(shape=(self.num_nodes, self.feature_dim,))

        # GCN model.
        #

        h_gcn = Dropout(0.3)(features)
        h_gcn = GCNLayer(64)([h_gcn, adj])
        h_gcn = BatchNormalization()(h_gcn)
        h_gcn = LeakyReLU()(h_gcn)
        y_gcn = GCNLayer(32)([h_gcn, adj])
        y_gcn = BatchNormalization()(y_gcn)
        y_gcn = LeakyReLU()(y_gcn)
        y_gcn = GCNLayer(self.latent_dim)([y_gcn, adj])
        y_gcn = BatchNormalization()(y_gcn)
        y_gcn = LeakyReLU()(y_gcn)
        y_gcn = Dropout(0.3)(y_gcn)

        output_feature = Flatten()(y_gcn)
        # output_feature = Lambda(lambda x: x[:, 0])(y_gcn)

        # Determine validity
        validity = Dense(1)(output_feature)
        label = Dense(self.num_classes + 1)(output_feature)

        model = Model([features, adj], [validity, label])
        model.summary()

        return model

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def wasserstein_sparsed_loss(self, y_true, y_pred):
        y_true_onehot = K.one_hot(y_true, self.num_classes+1)
        return K.mean(y_true_onehot * y_pred)

    def train(self, epochs, batch_size=128, sample_interval=50, d_weight=3, train_loss=0.5):

        # Get data
        X, A, y = load_data(path="../../data/cora/", dataset="cora")
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

        # Normalize X
        X = (X-np.repeat(0.5, X.shape[0]*X.shape[1]).reshape(X.shape))*2

        # filter
        print('Using local pooling filters...')
        support = 1
        graph = [X, A]

        d_loss = [1000]
        g_loss = [1000]

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            X_sample_unlabeled = []
            A_sample_unlabeled = []
            idx_real_unlabeled = []
            for _ in range(batch_size//2):
                x, a, i = self.sample_sub_graph(X, A, self.num_nodes, range(X.shape[0]))
                a = preprocess_adj(a)
                X_sample_unlabeled.append(x)
                A_sample_unlabeled.append(a)
                idx_real_unlabeled.append(i[0])
            for _ in range(batch_size//2):
                x, a, i = self.sample_sub_graph(X, A, self.num_nodes, idx_train)
                a = preprocess_adj(a)
                X_sample_unlabeled.append(x)
                A_sample_unlabeled.append(a)
                idx_real_unlabeled.append(i[0])
            X_sample_unlabeled = np.array(X_sample_unlabeled)
            A_sample_unlabeled = np.array(A_sample_unlabeled)

            print(idx_real_unlabeled)
            # print(A_sample_unlabeled)

            # Adversarial ground truths
            # valid = np.random.random((batch_size, 1))*0.5 + np.repeat(0.7, batch_size)
            # fake = np.random.random((batch_size, 1))*0.3
            valid = np.ones((batch_size, 1))
            fake = -np.ones((batch_size, 1))

            # Sample noise as generator input
            noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # node representation of
            sampled_labels = np.random.randint(1, self.num_classes, size=(batch_size, 1))

            # Generate a half batch of new images
            gen_graphs = self.generator.predict([sampled_labels, noise])

            # Node labels. 0-6 if image is valid or 7 if it is generated (fake)
            node_labels = np.array(y_train[idx_real_unlabeled])
            fake_labels = np.repeat(self.num_classes, batch_size)

            # Train the discriminator
            # if d_loss[0] > train_loss:
            if True:
                d_loss_real_unlabeled = self.discriminator_unlabeled.train_on_batch([X_sample_unlabeled, A_sample_unlabeled], [valid, node_labels])
                d_loss_fake_unlabeled = self.discriminator_unlabeled.train_on_batch(gen_graphs, [fake, fake_labels])
                d_loss = 0.5 * np.add(d_loss_real_unlabeled[:-1], d_loss_fake_unlabeled[:-1])

                for l in self.discriminator_unlabeled.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            # if g_loss[0] > train_loss:
            if epoch % d_weight == 0:
                g_loss = self.combined.train_on_batch([sampled_labels, noise], [valid, sampled_labels])

            if d_loss[0] <= train_loss and g_loss[0] <= train_loss:
                train_loss *= 0.75

            # Plot the progress

            print("{} [Disciminator loss: {}, validate acc:{} label acc:{}, generater loss: {}]".format(epoch, d_loss, d_loss[1]*100, d_loss[3]*100, g_loss))

            # print ("%d [Unlabeled loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss_unlabeled[0], 100*d_loss_unlabeled[3], 100*d_loss_unlabeled[4], g_loss[0]))
            # print("\t[Unlabeled loss real: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
            #     d_loss_real_unlabeled[0], 100 * d_loss_real_unlabeled[3], 100 * d_loss_real_unlabeled[4]))
            # print("\t[Unlabeled loss fake: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
            #     d_loss_fake_unlabeled[0], 100 * d_loss_fake_unlabeled[3], 100 * d_loss_fake_unlabeled[4]))
            #
            # print ("%d [Labeled loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            # print("\t[Labeled loss real: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
            #     d_loss_real[0], 100 * d_loss_real[3], 100 * d_loss_real[4]))
            # print("\t[Labeled loss fake: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
            #     d_loss_fake[0], 100 * d_loss_fake[3], 100 * d_loss_fake[4]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # self.save_model()
                self.val(y_val, idx_val, X, A, self.num_nodes)
                self.save_model()

    def sample_sub_graph(self, X, A, num_samles, from_idx=None):
        if from_idx:
            center_node_id = np.random.choice(from_idx, 1)[0]
        else:
            center_node_id = np.random.randint(0,X.shape[0],1)[0]
        return self.sample_sub_graph_with_center(X, A, num_samles, center_node_id, from_idx)

    def sample_sub_graph_with_center(self, X, A, num_samples, center_node_id, from_idx=None):
        A_dense = A.todense()
        X_ = np.zeros(shape=(num_samples, X.shape[1]))
        A_ = np.zeros(shape=(num_samples, num_samples))

        node_ids = self._sample_sub_graph_id(A_dense, num_samples, center_node_id, from_idx)

        for i in range(num_samples):
            if node_ids[i] < 0:
                break
            X_[i] = X[node_ids[i]]
            for j in range(num_samples):
                if node_ids[j] <0:
                    break
                A_[i][j] = A_dense[node_ids[i], node_ids[j]]

        return X_, A_, node_ids

    def _sample_sub_graph_id(self, A, num_samples, center_node_id, from_idx=None):
        node_ids = [(center_node_id,0)]
        if from_idx:
            tags = np.ones(A.shape[0])
            tags[from_idx] = 0
        else:
            tags = np.zeros(A.shape[0])
        tags[center_node_id] = 1

        count = 1
        for id_base,depth in node_ids:
            for id_next in range(A.shape[0]):
                if A[id_base, id_next] > 0 and tags[id_next] < 0.5 and depth < MAX_DEPTH:
                    node_ids.append((id_next, depth+1))
                    tags[id_next] = 1
                    count += 1
                    if count == num_samples:
                        return [ni[0] for ni in node_ids]

        [node_ids.append((-1,-1)) for i in range(count, num_samples)]
        return [ni[0] for ni in node_ids]

    def val(self, y_val, val_idx, X, A, batch_size):
        y_val = encode_onehot(y_val)
        pred_classes = []
        X_ = []
        A_ = []
        for idx in val_idx:
            x, a, _ = self.sample_sub_graph_with_center(X, A, batch_size, idx)
            a = preprocess_adj(a)
            X_.append(x)
            A_.append(a)
        X_ = np.array(X_)
        A_ = np.array(A_)
        pred_classes = self.discriminator_unlabeled.predict([X_, A_])[1]
        loss, acc = evaluate_preds(pred_classes, y_val, val_idx)
        print("Val: loss : %.2f,  acc : %.2f%%" % (loss, acc*100))

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])


        save(self.generator, "generator")
        save(self.discriminator_unlabeled, "discriminator_unlabeled")

if __name__ == '__main__':
    acgan = ACGAN(1433, 16, 7)
    acgan.train(epochs=14001, batch_size=8, sample_interval=200, d_weight=4, train_loss=2)
