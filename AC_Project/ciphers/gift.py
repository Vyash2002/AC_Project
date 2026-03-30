from core.utils import *
# =========================================================
# CIPHER 2 — GIFT-128
# =========================================================
_GIFT_SBOX = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9,
              0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
_GIFT_PERM = [4 * (i % 32) + (i // 32) for i in range(128)]

def _gift_rc_table() -> list:
    RC = []; c = 0b000001
    for _ in range(40):
        RC.append(c)
        fb = ((c >> 5) ^ (c >> 4) ^ (c >> 2) ^ (c >> 1)) & 1
        c  = ((c << 1) | fb) & 0x3F
    return RC

_GIFT_RC = _gift_rc_table()

def gift_encrypt(P: bytes, rounds: int, key: bytes, **_) -> bytes:
    state   = int.from_bytes(P,   'big')
    key_int = int.from_bytes(key, 'big')
    U = (key_int >> 64) & MASK64
    V =  key_int        & MASK64
    M32 = 0xFFFF_FFFF
    for rnd in range(rounds):
        # SubCells
        ns = 0
        for nib in range(32):
            sh = (31 - nib) * 4
            ns |= _GIFT_SBOX[(state >> sh) & 0xF] << sh
        state = ns
        # PermBits
        ns = 0
        for i in range(128):
            ns |= ((state >> (127 - i)) & 1) << (127 - _GIFT_PERM[i])
        state = ns
        # AddRoundKey
        state ^= ((V >> 32) & M32) << 64 | (V & M32)
        rc = _GIFT_RC[rnd]
        for b in range(6):
            if (rc >> b) & 1:
                state ^= 1 << (23 - 4 * b)
        # Key update
        U_lo = U & M32; U_hi = (U >> 32) & M32
        U = (((U_hi >> 12) | (U_hi << 20)) & M32) << 32 | ((U_lo >> 2) | (U_lo << 30)) & M32
        V = ((V >> 12) | (V << 52)) & MASK64
    return state.to_bytes(16, 'big')
