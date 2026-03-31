from core.utils import *
# =========================================================
# CIPHER 6 — RECTANGLE-128  (SPN, Zhang et al. 2015)
# =========================================================
# Bit-sliced 4×16 state; 4-bit S-box column-wise; row-shift permutation.
# Two independent 64-bit blocks → 128-bit output (same as PRESENT).
_RECT_SBOX = [0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9,
              0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2]
_RECT_RC   = [0x01,0x02,0x04,0x08,0x10,0x20,0x03,0x06,0x0C,0x18,0x30,
              0x21,0x05,0x0A,0x14,0x28,0x11,0x22,0x07,0x0E,0x1C,0x38,
              0x31,0x23,0x09,0x12,0x24,0x0B,0x16,0x2C,0x19,0x32,0x25,
              0x0D,0x1A,0x34,0x29,0x13,0x26,0x0F]

def rectangle_encrypt(P: bytes, rounds: int, key: bytes, **_) -> bytes:
    def _half(block: int, rounds: int, k: int) -> int:
        W0 = (k >> 64) & MASK64
        W1 =  k        & MASK64
        subkeys = []
        for r in range(rounds + 1):
            subkeys.append(W0)
            W0 = ((W0 << 8) | (W0 >> 56)) & MASK64
            top = (W0 >> 60) & 0xF
            W0  = (W0 & ~(0xF << 60)) | (_RECT_SBOX[top] << 60)
            if r < len(_RECT_RC):
                W0 ^= _RECT_RC[r]
            W0, W1 = W1, W0

        def unpack(s):
            rows = [0] * 4
            for col in range(16):
                for row in range(4):
                    rows[row] |= ((s >> (col * 4 + row)) & 1) << col
            return rows

        def pack(rows):
            s = 0
            for col in range(16):
                for row in range(4):
                    s |= ((rows[row] >> col) & 1) << (col * 4 + row)
            return s

        state = block & MASK64
        sigma = [0, 1, 12, 13]
        for r in range(rounds):
            state ^= subkeys[r]
            rows = unpack(state)
            for col in range(16):
                nib = sum(((rows[row] >> col) & 1) << row for row in range(4))
                out = _RECT_SBOX[nib]
                for row in range(4):
                    rows[row] = (rows[row] & ~(1 << col)) | (((out >> row) & 1) << col)
            for i in range(4):
                s = sigma[i]
                if s:
                    rows[i] = ((rows[i] << s) | (rows[i] >> (16 - s))) & 0xFFFF
            state = pack(rows)
        state ^= subkeys[rounds]
        return state

    k_int  = int.from_bytes(key, 'big')
    lo     = int.from_bytes(P[:8], 'big')
    hi     = int.from_bytes(P[8:], 'big')
    out_lo = _half(lo, rounds, k_int)
    out_hi = _half(hi, rounds, k_int ^ 0xCAFE_BABE_CAFE_BABE)
    return out_lo.to_bytes(8, 'big') + out_hi.to_bytes(8, 'big')
