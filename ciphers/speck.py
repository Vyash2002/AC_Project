from core.utils import *
# =========================================================
# CIPHER 4 — SPECK-128/128  (ARX Feistel, NSA 2013)
# =========================================================
# Round function: x = (rotr(x,8) + y) ^ k;  y = rotl(y,3) ^ x
def speck_encrypt(P: bytes, rounds: int, key: bytes, **_) -> bytes:
    x = int.from_bytes(P[:8],   'big')
    y = int.from_bytes(P[8:],   'big')
    b = int.from_bytes(key[:8], 'big')
    a = int.from_bytes(key[8:], 'big')
    subkeys = [b]
    MOD = 1 << 64
    for i in range(rounds - 1):
        a = (rotr64(a, 8) + b) & MASK64 ^ i
        b = rotl64(b, 3) ^ a
        subkeys.append(b)
    for k_i in subkeys[:rounds]:
        x = (rotr64(x, 8) + y) % MOD ^ k_i
        y = rotl64(y, 3) ^ x
    return x.to_bytes(8, 'big') + y.to_bytes(8, 'big')
