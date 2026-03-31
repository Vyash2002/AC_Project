import os
import random
import numpy as np

from core.utils import xor_bytes, bytes_to_bits
from core.config import BLOCK_SIZE, CIPHER_DELTA
from core.dispatch import encrypt

def generate_dataset(cipher, rounds, num_samples, data_dir):
    delta = CIPHER_DELTA[cipher]

    X1, X2, Y = [], [], []

    for _ in range(num_samples):
        key   = os.urandom(16)
        nonce = os.urandom(16)
        P1    = os.urandom(BLOCK_SIZE)
        P2    = xor_bytes(P1, delta)

        if random.random() < 0.5:
            C1 = encrypt(cipher, P1, rounds, key, nonce)
            C2 = encrypt(cipher, P2, rounds, key, nonce)
            label = 1
        else:
            C1 = os.urandom(BLOCK_SIZE)
            C2 = os.urandom(BLOCK_SIZE)
            label = 0

        X1.append(bytes_to_bits(P1) + bytes_to_bits(C1))
        X2.append(bytes_to_bits(P2) + bytes_to_bits(C2))
        Y.append(label)

    tag = f"{cipher}_r{rounds}"

    np.save(os.path.join(data_dir, f"X1_{tag}.npy"), np.array(X1, dtype=np.float32))
    np.save(os.path.join(data_dir, f"X2_{tag}.npy"), np.array(X2, dtype=np.float32))
    np.save(os.path.join(data_dir, f"y_{tag}.npy"),  np.array(Y, dtype=np.int8))