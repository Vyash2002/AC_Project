from core.utils import *

# =========================================================
# CIPHER 1 — ASCON-128
# =========================================================
def _ascon_permutation(S: list, rounds: int) -> list:
    RC = [0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5,
          0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b]
    S = list(S)
    for r in range(12 - rounds, 12):
        S[2] ^= RC[r]
        x0, x1, x2, x3, x4 = S
        x0 ^= x4; x4 ^= x3; x2 ^= x1
        t0 = (~x0) & x1 & MASK64
        t1 = (~x1) & x2 & MASK64
        t2 = (~x2) & x3 & MASK64
        t3 = (~x3) & x4 & MASK64
        t4 = (~x4) & x0 & MASK64
        x0 ^= t1; x1 ^= t2; x2 ^= t3; x3 ^= t4; x4 ^= t0
        x1 ^= x0; x0 ^= x4; x3 ^= x2; x2 = (~x2) & MASK64
        S = [x0, x1, x2, x3, x4]
        S[0] ^= rotr64(S[0], 19) ^ rotr64(S[0], 28)
        S[1] ^= rotr64(S[1], 61) ^ rotr64(S[1], 39)
        S[2] ^= rotr64(S[2],  1) ^ rotr64(S[2],  6)
        S[3] ^= rotr64(S[3], 10) ^ rotr64(S[3], 17)
        S[4] ^= rotr64(S[4],  7) ^ rotr64(S[4], 41)
    return S

def ascon_encrypt(P: bytes, rounds: int, key: bytes, nonce: bytes) -> bytes:
    IV  = 0x80400c0600000000
    K   = [int.from_bytes(key[i:i+8],   'big') for i in range(0, 16, 8)]
    N   = [int.from_bytes(nonce[i:i+8], 'big') for i in range(0, 16, 8)]
    S   = [IV, K[0], K[1], N[0], N[1]]
    Pw  = [int.from_bytes(P[i:i+8],     'big') for i in range(0, 16, 8)]
    S[0] ^= Pw[0]; S[1] ^= Pw[1]
    S = _ascon_permutation(S, rounds)
    return b''.join(w.to_bytes(8, 'big') for w in [S[0], S[1]])
