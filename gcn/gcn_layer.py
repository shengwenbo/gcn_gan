from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import copy


class GCNLayer(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = list(features_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)  # (batch_size, num_nodes, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        input_dim = features_shape[-1]

        self.Q = self.add_weight(shape=(input_dim, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.W = self.add_weight(shape=(input_dim*2, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.q = self.add_weight(shape=(input_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.w = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.q = None
            self.w = None
        self.built = True

    def call(self, inputs, **kwargs):
        features = inputs[0]
        adj = inputs[1]
        num_nodes = features.shape[1]

        features_dense = K.dot(features, self.Q)
        if self.q:
            features_dense += self.q

        features_agg = K.batch_dot(adj, features_dense)

        features_conc = K.concatenate([features, features_agg])

        features_new = K.dot(features_conc, self.W)
        if self.w:
            features_new += self.w

        features_new = K.l2_normalize(features_new)

        return self.activation(features_new)
