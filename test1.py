from random import randint
import numpy as np
import typing

import tensorflow as tf
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent


def tfPrint(d, T): return tf.Print(input_=T, data=[T, tf.shape(T)], message=d)


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


def generate_sequence(length: int, n_unique: int) -> list:
    return [randint(0, n_unique - 1) for _ in range(length)]


sequence = generate_sequence(5, 50)


def one_hot_encode(sequence: list, n_unique: int) -> np.array:
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)


def one_hot_decode(encoded_seq: np.array) -> list:
    return [np.argmax(vector) for vector in encoded_seq]


def get_pair(n_in: int, n_out: int, n_unique: int) -> typing.Tuple[np.array, np.array]:
    # generate random sequence
    sequence_in = generate_sequence(n_in, n_unique)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]

    # one hot encode
    X = one_hot_encode(sequence_in, n_unique)
    y = one_hot_encode(sequence_out, n_unique)

    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))

    return (X, y)


X, y = get_pair(5, 2, 50)

n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2

# model architecture
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(n_features, activation='softmax')))

# Summary
print(model.summary())

# compile
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

for epoch in range(5000):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    model.fit(X, y)

# Evaluate
total, correct = 100, 0
for _ in range(total):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    preds = model.predict(X, verbose=0)

    print('Expected = {}, Predicted = {}'.format(
        one_hot_decode(y[0]), one_hot_decode(preds[0])))

    if np.array_equal(one_hot_decode(y[0]), one_hot_decode(preds[0])):
        correct += 1

print('Accuracy  = {}'.format(correct * 100 / total))


class AttentionDecoder(Recurrent):

    def __init__(self,
                 units,
                 output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an attentiondecoder that takes in a sequence
        encoded by a encoder and outputs the decoded state
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activations)
        self.kernel_initizalizer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True



        
