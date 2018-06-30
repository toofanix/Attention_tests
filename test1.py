from random import randint
import numpy as np
import typing


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


def get_pair(n_in: int, n_out: int, n_unique: int) -> typing.Any:
	# generate random sequence
	sequence_in = generate_sequence(n_in, n_unique)
	sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]

	# one hot encode
	X = one_hot_encode(sequence_in, n_unique)
	y = one_hot_encode(sequence_out, n_unique)

	# reshape as 3D

	return sequence_out