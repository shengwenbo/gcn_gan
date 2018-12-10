from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate, subtract, add, dot, Lambda, RepeatVector
from keras.layers.recurrent import LSTM
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

from gcn.gcn_layer import GCNLayer

from utils import *

import os

class ACGAN():
    def __init__(self, feature_dim, num_nodes, num_classes, latent_dim=32):
        # Input shape
        self.latent_dim = latent_dim

        self.feature_dim = feature_dim
        self.num_nodes = num_nodes
        self.num_classes = num_classes

        optimizer = Adam(0.0001, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        self.discriminator_labeled = self.build_discriminator_labeled()
        self.discriminator_labeled.compile(loss=["sparse_categorical_crossentropy"],
            optimizer=optimizer,
            metrics=['accuracy'])

        # load saved model
        if(os.listdir("saved_model")):
            print("Loading weights from saved_model/")
            self.discriminator_labeled.load_weights("saved_model/discriminator_labeled_weights.hdf5")
            print("Done.")

    def build_discriminator_labeled(self):

        adj = Input(shape=(self.num_nodes, self.num_nodes,))
        features = Input(shape=(self.num_nodes, self.feature_dim,))

        # GCN model.
        #
        h_gcn = Dropout(0.5)(features)
        h_gcn = GCNLayer(32)([h_gcn, adj])
        h_gcn = LeakyReLU()(h_gcn)
        h_gcn = Dropout(0.5)(h_gcn)
        y_gcn = GCNLayer(self.latent_dim)([h_gcn, adj])
        y_gcn = LeakyReLU()(y_gcn)

        # y_gcn = Flatten()(y_gcn)
        # output_feature = Dense(self.latent_dim, name="graph_embedding")(y_gcn)

        output_feature = Lambda(lambda x: x[:, 0])(y_gcn)

        label = Dense(self.num_classes+1, activation="softmax")(output_feature)

        model = Model([features, adj], [label])
        model.summary()

        return model

        return model

    def train(self, epochs, batch_size=128, sample_interval=50, d_weight=3):

        # Get data
        X, A, y = load_data(path="../../data/cora/", dataset="cora")
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

        # Normalize X
        X /= X.sum(1).reshape(-1, 1)

        # filter
        print('Using local pooling filters...')
        support = 1
        graph = [X, A]

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            X_sample_labeled = []
            A_sample_labeled = []
            idx_real_labeled = []
            for i in range(batch_size):
                x, a, i = self.sample_sub_graph(X, A, self.num_nodes, idx_train)
                a = preprocess_adj(a)
                X_sample_labeled.append(x)
                A_sample_labeled.append(a)
                idx_real_labeled.append(i[0])
            X_sample_labeled = np.array(X_sample_labeled)
            A_sample_labeled = np.array(A_sample_labeled)

            # Adversarial ground truths
            valid = np.ones((self.num_nodes, 1))

            # Node labels. 0-6 if image is valid or 7 if it is generated (fake)
            node_labels = y_train[idx_real_labeled]

            d_loss_labeled = self.discriminator_labeled.train_on_batch([X_sample_labeled, A_sample_labeled], [node_labels])

            # Plot the progress

            print("{} [loss: {}]".format(epoch,d_loss_labeled))

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
        X_ = []
        A_ = []
        for idx in val_idx:
            x, a, _ = self.sample_sub_graph_with_center(X, A, batch_size, idx)
            a = preprocess_adj(a)
            X_.append(x)
            A_.append(a)
        X_ = np.array(X_)
        A_ = np.array(A_)
        pred_classes = self.discriminator_labeled.predict([X_, A_])
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

        save(self.discriminator_labeled, "discriminator_labeled")

if __name__ == '__main__':
    acgan = ACGAN(1433, 16, 7)
    acgan.train(epochs=14001, batch_size=32, sample_interval=200, d_weight=4)
