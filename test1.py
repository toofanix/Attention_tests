from random import randint
import numpy as np

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

