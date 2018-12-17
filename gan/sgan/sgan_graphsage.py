from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate, subtract, add, dot, Lambda, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from scipy.sparse import csr_matrix

from tensorflow import boolean_mask

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import pickle as pk

from gcn.graphsage_layer import GraphSageLayer

from utils import *

import os

MAX_DEPTH = 3
GCN_LAYERS = 1

class SGAN():
    def __init__(self, feature_dim, num_neighbors, num_classes, latent_dim=32, dropout_rate=0., mlp_units=(32, 64, 128)):
        # Input shape
        self.latent_dim = latent_dim

        self.feature_dim = feature_dim
        self.num_neighbors = num_neighbors
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.mlp_units = mlp_units

        optimizer = Adam(0.001, 0.5)
        losses = [neg_categorical_crossentropy] + ['mse']*GCN_LAYERS
        loss_weights = [1] + [0.01]*GCN_LAYERS

        # Build and compile the discriminator
        self.discriminator_base = self.build_discriminator()

        x = Input((self.feature_dim,))
        neighbors = Input((self.num_neighbors, self.feature_dim))
        label_pred = self.discriminator_base([x, neighbors])[0]
        self.discriminator_l = Model([x, neighbors], label_pred)
        self.discriminator_l.compile(loss='categorical_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.discriminator_u = Model([x, neighbors], label_pred)
        self.discriminator_u.compile(loss=neg_categorical_crossentropy,
                                    optimizer=optimizer,
                                    metrics=["accuracy"])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))

        [x, gen_neighbors] = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator_base.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image

        out = self.discriminator_base([x, gen_neighbors])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(noise, out)
        self.combined.compile(loss=losses,
                              loss_weights=loss_weights,
            optimizer=optimizer)
        print("\n\n===================================combined==============================")
        self.combined.summary()

        # load saved model
        if(os.listdir("saved_model")):
            print("Loading weights from saved_model/")
            self.generator.load_weights("saved_model/generator_weights.hdf5")
            self.discriminator_base.load_weights("saved_model/discriminator_weights.hdf5")
            print("Done.")

    def build_generator(self):
        """
        directly concat label info to feature info
        :return:
        """
        noise = Input(shape=(self.latent_dim,))

        # generative model.
        #
        # MLP
        mlp = noise
        for u in self.mlp_units:
            mlp = Dense(u, activation="relu")(mlp)
            mlp = Dropout(self.dropout_rate)(mlp)

        # generate center node
        x = Dense(self.feature_dim,)(mlp)
        x = Activation("tanh")(x)

        # generate neighbors
        neighbors = Reshape((self.num_neighbors, self.feature_dim))(Dense(self.num_neighbors * self.feature_dim, )(mlp))
        neighbors = Activation("tanh")(neighbors)

        # Compile model
        model = Model(inputs=noise, outputs=[x, neighbors])
        print("\n\n===================================generator==============================")
        model.summary()

        return model

    def build_discriminator(self):

        x = Input(shape=(self.feature_dim,))
        neighbors = Input(shape=(self.num_neighbors, self.feature_dim))

        layer_features = []

        # GCN model.
        #
        x_drop = Dropout(self.dropout_rate)(x)
        neighbors_drop = Dropout(self.dropout_rate)(neighbors)

        y_gcn = GraphSageLayer(512, self.num_neighbors)([x_drop, neighbors_drop])
        layer_features.append(y_gcn)
        y_gcn = LeakyReLU()(y_gcn)

        label = Dense(self.num_classes + 1, activation="softmax")(y_gcn)

        model = Model([x, neighbors], [label]+layer_features)
        print("\n\n===================================discriminator==============================")
        model.summary()

        return model

    def train(self, epochs, batch_size=128, sample_interval=50, g_weight=1, d_weight=3, u_weight=5, train_loss=0.5):

        # Get data
        X, A, y = load_data(path="../../data/cora/", dataset="cora")
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

        Neighbors = self.prepare_neighbors(X,A)

        # Normalize X
        # X /= X.sum(1).reshape(-1, 1)
        X = (X-np.full_like(X, 0.5))*2

        d_loss = [1000]
        g_loss = [1000]

        fake_labels = np.concatenate(
            [np.zeros((batch_size, self.num_classes)), np.ones((batch_size, 1))], axis=-1)
        un_labels = np.concatenate(
            [np.ones((batch_size, self.num_classes)), np.zeros((batch_size, 1))], axis=-1)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if epoch % u_weight == 0:
                supervised = False
            else:
                supervised = False

            X_sample = []
            N_sample = []
            idx_sample = []
            if supervised:
                for _ in range(batch_size):
                    x, n, i = self.sample_neighbors(X, Neighbors, self.num_neighbors, idx_train)
                    X_sample.append(x)
                    N_sample.append(n)
                    idx_sample.append(i)
            else:
                for _ in range(batch_size):
                    x, n, i = self.sample_neighbors(X, Neighbors, self.num_neighbors, range(X.shape[0]))
                    X_sample.append(x)
                    N_sample.append(n)
                    idx_sample.append(i)

            X_sample = np.array(X_sample)
            X_sample = np.reshape(X_sample, (-1, self.feature_dim))
            N_sample = np.array(N_sample)

            print(idx_sample)

            # Adversarial ground truths
            # valid = np.random.random((batch_size, 1))*0.5 + np.repeat(0.7, batch_size)
            # fake = np.random.random((batch_size, 1))*0.3

            # Sample noise as generator input
            noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))

            # Node labels. 0-6 if image is valid or 7 if it is generated (fake)
            if supervised:
                node_labels = convert_to_one_hot(y_train[idx_sample], self.num_classes+1)
            else:
                node_labels = fake_labels

            # Train the discriminator
            if epoch % (g_weight + d_weight) < d_weight:
                gen_neighbors = self.generator.predict(noise)
                if supervised:
                    d_loss_real = self.discriminator_l.train_on_batch([X_sample, N_sample], node_labels)
                else:
                    d_loss_real = self.discriminator_u.train_on_batch([X_sample, N_sample], node_labels)
                d_loss_fake = self.discriminator_l.train_on_batch(gen_neighbors, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            if epoch % (g_weight + d_weight) >= d_weight:
                layer_out = self.discriminator_base.predict([X_sample, N_sample])[1:]
                g_loss = self.combined.train_on_batch(noise, [fake_labels]+layer_out)

            if d_loss[0] <= train_loss and g_loss[0] <= train_loss:
                train_loss *= 0.75

            # Plot the progress

            print("{} [Disciminator loss: {}, generater loss: {}]".format(epoch, d_loss, g_loss))

            # print ("%d [Unlabeled loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss_unlabeled[0], 100*d_loss_unlabeled[3], 100*d_loss_unlabeled[4], g_loss[0]))
            # print("\t[Unlabeled loss real: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
            #     d_loss_real[0], 100 * d_loss_real[3], 100 * d_loss_real[4]))
            # print("\t[Unlabeled loss fake: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
            #     d_loss_fake[0], 100 * d_loss_fake[3], 100 * d_loss_fake[4]))
            #
            # print ("%d [Labeled loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            # print("\t[Labeled loss real: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
            #     d_loss_real[0], 100 * d_loss_real[3], 100 * d_loss_real[4]))
            # print("\t[Labeled loss fake: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
            #     d_loss_fake[0], 100 * d_loss_fake[3], 100 * d_loss_fake[4]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # self.save_model()
                self.val(y_val, idx_val, X, Neighbors, self.num_neighbors)
                self.save_model()

    def prepare_neighbors(self, X, A, data_path="../../data/cora/neighbors.pkl"):
        print("Preparing neighbor info...")
        if os.path.exists(data_path):
            with open(data_path, "rb") as fin:
                ret = pk.load(fin)
            return ret

        ret = []
        for center_node_id in range(X.shape[0]):
            neighbors = [id for id in range(X.shape[0]) if A[center_node_id, id] > 0.1]
            ret.append(neighbors)

        with open(data_path, "wb") as fout:
            pk.dump(ret, fout)

        print("Done.")
        return ret

    def sample_neighbors(self, X, Neighbors, num_neighbors, from_idx=None):
        if from_idx:
            center_node_id = np.random.choice(from_idx, 1)[0]
        else:
            center_node_id = np.random.randint(0,X.shape[0],1)[0]
        return self.sample_neighbors_with_center(X, Neighbors, num_neighbors, center_node_id)

    def sample_neighbors_with_center(self, X, Neighbors, num_neighbors, center_node_id):
        neibors = Neighbors[center_node_id]
        if len(neibors) > num_neighbors:
            neibors = np.random.choice(neibors, num_neighbors, replace=False)
        else:
            neibors = np.random.choice(neibors, num_neighbors, replace=True)
        neibors_feature = X[neibors]
        x = X[center_node_id]
        return x, neibors_feature, center_node_id

    def val(self, y_val, val_idx, X, Neighbors, batch_size):
        y_val = encode_onehot(y_val)
        pred_classes = []
        X_ = []
        N_ = []
        for idx in val_idx:
            x, n, _ = self.sample_neighbors_with_center(X, Neighbors, batch_size, idx)
            X_.append(x)
            N_.append(n)
        X_ = np.array(X_)
        X_ = np.reshape(X_, (-1, self.feature_dim))
        N_ = np.array(N_)
        pred_classes = self.discriminator_l.predict([X_, N_])
        loss, acc = evaluate_preds(pred_classes, y_val, val_idx)
        print("Val: loss : %.2f,  acc : %.2f%%" % (loss, acc*100))

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
        save(self.discriminator_base, "discriminator")

if __name__ == '__main__':
    acgan = SGAN(1433, 16, 7)
    acgan.train(epochs=14001, batch_size=32, sample_interval=200, g_weight=10, d_weight=1, u_weight=10, train_loss=3)
