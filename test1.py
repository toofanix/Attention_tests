from random import randint
import numpy as np
import typing

from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential


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
print (model.summary())
# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

