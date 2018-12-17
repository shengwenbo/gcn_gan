from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import copy


class GraphSageLayer(Layer):
    def __init__(self, units,
                 num_neighbors,
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
        super(GraphSageLayer, self).__init__(**kwargs)
        self.units = units
        self.num_neighbors = num_neighbors
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
                                      name='kernel_Q',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.W = self.add_weight(shape=(input_dim*3, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel_W',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.q = self.add_weight(shape=(input_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias_q',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.w = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias_w',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.q = None
            self.w = None
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs[0]
        neighbors = inputs[1]

        fc_neigh = K.dot(neighbors, self.Q)
        if self.use_bias:
            fc_neigh += self.q

        avg_pool = K.sum(fc_neigh, axis=1) / self.num_neighbors
        max_pool = K.max(fc_neigh, axis=1)
        pool = K.concatenate([avg_pool, max_pool])

        x = K.concatenate([x, pool])
        y = K.dot(x, self.W)
        if self.use_bias:
            y += self.w

        return self.activation(y)
