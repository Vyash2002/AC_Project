from core.utils import *
# =========================================================
# CIPHER 8 — SPARX-128/128  (ARX-based, Dinu et al. 2016)
# =========================================================
# SPARX is an ARX cipher using a Feistel-like structure with
# a strong ARX box (A-box). It combines ideas from SPECK but
# applies them in a parallel structure for improved diffusion.
#
# Structure:
#   - State: 4 x 32-bit words
#   - Round: ARX-box on each branch + linear mixing layer
#
# Reference: Dinu et al., "SPARX: a Family of ARX Block Ciphers"

def sparx_arx_box(x, y):
    """SPARX A-box (32-bit words)"""
    x = (rotl32(x, 9) + y) & 0xFFFFFFFF
    y = rotl32(y, 2) ^ x
    return x, y

def sparx_linear_layer(x0, x1, x2, x3):
    """Linear diffusion layer"""
    t = x0 ^ x1 ^ x2 ^ x3
    t = rotl32(t, 8)
    return x0 ^ t, x1 ^ t, x2 ^ t, x3 ^ t

def sparx_encrypt(P: bytes, rounds: int, key: bytes, nonce: None) -> bytes:
    """
    Simplified SPARX-128/128:
    - 128-bit block (4 x 32-bit words)
    - ARX-based structure with parallel branches
    """

    # Split plaintext
    x0 = int.from_bytes(P[0:4], 'big')
    x1 = int.from_bytes(P[4:8], 'big')
    x2 = int.from_bytes(P[8:12], 'big')
    x3 = int.from_bytes(P[12:16], 'big')

    # Key split (4 x 32-bit)
    k = [int.from_bytes(key[i:i+4], 'big') for i in range(0, 16, 4)]

    for r in range(rounds):
        # Add round key (simple schedule)
        x0 ^= k[0]
        x1 ^= k[1]
        x2 ^= k[2]
        x3 ^= k[3]

        # ARX boxes (two parallel branches)
        x0, x1 = sparx_arx_box(x0, x1)
        x2, x3 = sparx_arx_box(x2, x3)

        # Linear mixing layer
        x0, x1, x2, x3 = sparx_linear_layer(x0, x1, x2, x3)

        # Simple key schedule (rotate words)
        k = k[1:] + k[:1]

    # Final whitening
    x0 ^= k[0]
    x1 ^= k[1]
    x2 ^= k[2]
    x3 ^= k[3]

    return (
        x0.to_bytes(4, 'big') +
        x1.to_bytes(4, 'big') +
        x2.to_bytes(4, 'big') +
        x3.to_bytes(4, 'big')
    )
