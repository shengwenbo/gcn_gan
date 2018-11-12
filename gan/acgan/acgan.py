from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate, subtract, add, dot, Lambda
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

from gcn.graph import GraphConvolution

from utils import *

import os

class ACGAN():
    def __init__(self, feature_shape, num_classes):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.feature_dim = feature_shape[1]
        self.num_nodes = feature_shape[0]
        self.num_classes = num_classes

        optimizer = Adam(0.001, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator_merge()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        G = Input(shape=(None, None), batch_shape=(None, None), sparse=True)
        Noise = Input(shape=(self.latent_dim,))
        Label = Input(shape=(1,))
        Node = self.generator([Noise, Label, G])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image

        valid, target_label = self.discriminator([Node, G])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([Noise, Label, G], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

        # load saved model
        if(os.listdir("saved_model")):
            print("Loading weights from saved_model/")
            self.generator.load_weights("saved_model/generator_weights.hdf5")
            self.discriminator.load_weights("saved_model/discriminator_weights.hdf5")
            print("Done.")

    def build_generator_merge(self):
        """
        directly concat label info to feature info
        :return:
        """
        G = Input(shape=(None, None), batch_shape=(None, None), sparse=True)
        Noise = Input(shape=(self.latent_dim,))
        Label = Input(shape=(1,))
        Label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(Label))
        Label_in = multiply([Noise, Label_embedding])

        # GCN model.
        #
        H_gcn = Dropout(0.5)(Label_in)
        H_gcn = GraphConvolution(16, 1, kernel_regularizer=l2(5e-4))([H_gcn]+[G])
        H_gcn = LeakyReLU()(H_gcn)
        H_gcn = Dropout(0.5)(H_gcn)
        Y_gcn = GraphConvolution(self.feature_dim, 1, activation='softmax')([H_gcn]+[G])

        # Compile model
        model = Model(inputs=[Noise, Label, G], outputs=Y_gcn)
        model.summary()

        return model

    def build_discriminator(self):

        G = Input(shape=(None, None), batch_shape=(None, None), sparse=True)
        X_in = Input(shape=(self.feature_dim, ))

        # GCN model.
        #
        H_gcn = Dropout(0.5)(X_in)
        H_gcn = GraphConvolution(16, 1, kernel_regularizer=l2(5e-4))([H_gcn, G])
        H_gcn = LeakyReLU()(H_gcn)
        H_gcn = Dropout(0.5)(H_gcn)
        Y_gcn = GraphConvolution(self.feature_dim, 1, activation='softmax')([H_gcn, G])

        features = Y_gcn

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        model = Model([X_in, G], [validity, label])
        model.summary()

        return model

    def train(self, epochs, batch_size=128, sample_interval=50, d_weight=3):

        # Get data
        X, A, y = load_data(path="../../data/cora/", dataset="cora")
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

        # Normalize X
        X /= X.sum(1).reshape(-1, 1)

        # filter
        print('Using local pooling filters...')
        A_ = preprocess_adj(A, True)
        support = 1
        graph = [X, A]

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            X_sample, A_sample, idx_real = self.sample_sub_graph(X, A, batch_size, idx_train)
            _, A_sample_gen, idx_gen = self.sample_sub_graph(X, A, batch_size)

            A_sample = preprocess_adj(A_sample, True)
            A_sample_gen = preprocess_adj(A_sample_gen, True)

            # Adversarial ground truths
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Sample noise as generator input
            noise = np.random.normal(0, 1, size=(batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # node representation of
            sampled_labels = np.random.randint(1, self.num_classes, size=(batch_size, 1))

            # Generate a half batch of new images
            gen_graph = self.generator.predict([noise, sampled_labels, A_sample_gen], batch_size=batch_size)

            # Node labels. 0-6 if image is valid or 7 if it is generated (fake)
            node_labels = y_train[idx_real]
            fake_labels = np.repeat(self.num_classes, batch_size)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([X_sample, A_sample], [valid, node_labels])#, sample_weight=[valid_mask, valid_mask])
            d_loss_fake = self.discriminator.train_on_batch([gen_graph, A_sample_gen], [fake, fake_labels])#, sample_weight=[gen_mask[:, 0], gen_mask[:, 0]])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            if epoch % d_weight == 0:
                g_loss = self.combined.train_on_batch([noise, sampled_labels, A_sample_gen], [valid, sampled_labels])#, sample_weight=gen_mask[:0])

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            print("\t[D loss real: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
                d_loss_real[0], 100 * d_loss_real[3], 100 * d_loss_real[4]))
            print("\t[D loss fake: %f, acc.: %.2f%%, op_acc: %.2f%%]" % (
                d_loss_fake[0], 100 * d_loss_fake[3], 100 * d_loss_fake[4]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # self.save_model()
                self.val(y_val, idx_val, X, A_, batch_size)
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

        return X_, csr_matrix(A_), node_ids

    def _sample_sub_graph_id(self, A, num_samples, center_node_id, from_idx=None):
        node_ids = [center_node_id]
        if from_idx:
            tags = np.ones(A.shape[0])
            tags[from_idx] = 0
        else:
            tags = np.zeros(A.shape[0])
        tags[center_node_id] = 1

        count = 1
        for id_base in node_ids:
            for id_next in range(A.shape[0]):
                if A[id_base, id_next] > 0 and tags[id_next] < 0.5:
                    node_ids.append(id_next)
                    tags[id_next] = 1
                    count += 1
                    if count == num_samples:
                        return node_ids

        [node_ids.append(-1) for i in range(count, num_samples)]
        return node_ids

    def val(self, y_val, val_idx, X, A, batch_size):
        y_val = encode_onehot(y_val)
        pred_classes = []
        for idx in val_idx:
            X_, A_, _ = self.sample_sub_graph_with_center(X, A, batch_size, idx)
            pred_classes.append(self.discriminator.predict([X_, A_], batch_size=batch_size)[1][0])
        pred_classes = np.array(pred_classes)
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
        save(self.discriminator, "discriminator")

if __name__ == '__main__':
    acgan = ACGAN((2708, 1433), 7)
    acgan.train(epochs=14001, batch_size=16, sample_interval=200, d_weight=4)
