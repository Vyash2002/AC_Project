from core.utils import *

# =========================================================
# CIPHER 3 — SIMON-128/128  (Feistel, NSA 2013)
# =========================================================
# Round function: f(x) = (rotl(x,1) AND rotl(x,8)) XOR rotl(x,2)
# Key schedule (m=2): k[i] = c ^ z_bit ^ k[i-2]
#                             ^ rotr(k[i-1],3) ^ rotr(k[i-1],4)
#
# FIX: z2 must be exactly 62 bits (the period of the sequence).
# The original code had a 65-bit literal (three extra leading bits),
# causing wrong key material after position 62 due to the % 62 wrap.
_SIMON_C  = 0xFFFF_FFFF_FFFF_FFFC   # c = 2^64 - 4
# z2 sequence, period 62, LSB-first (NSA spec Table 4) — exactly 62 bits:
_SIMON_Z2 = 0b10111000011001110100010000100101001111010010010001100001111101

def simon_encrypt(P: bytes, rounds: int, key: bytes, **_) -> bytes:
    x  = int.from_bytes(P[:8], 'big')
    y  = int.from_bytes(P[8:], 'big')
    k0 = int.from_bytes(key[:8], 'big')
    k1 = int.from_bytes(key[8:], 'big')
    subkeys = [k0, k1]
    for i in range(2, rounds):
        tmp    = rotr64(subkeys[-1], 3)
        tmp   ^= rotr64(tmp, 1)
        z_bit  = (_SIMON_Z2 >> ((i - 2) % 62)) & 1
        tmp   ^= _SIMON_C ^ z_bit ^ subkeys[-2]
        subkeys.append(tmp)
    for i in range(rounds):
        f    = (rotl64(x, 1) & rotl64(x, 8)) ^ rotl64(x, 2)
        x, y = y ^ f ^ subkeys[i], x
    return x.to_bytes(8, 'big') + y.to_bytes(8, 'big')
