from core.utils import *
# =========================================================
# CIPHER 9 — XOODYAK  (Daemen et al. 2018)
# =========================================================
# Built on the 384-bit Xoodoo permutation (3 planes × 4 lanes × 32 bits).
# State layout: A[4*y + x] = a[y][x], y in {0,1,2}, x in {0,1,2,3}
#
# One Xoodoo round — exactly 5 steps per the official spec:
#   θ      : P[x] = A[x,0]^A[x,1]^A[x,2]
#             E[x] = rotl(P[x-1],5) ^ rotl(P[x-1],14)
#             A[x,y] ^= E[x]
#   ρ_west : A[x,1] = A[x-1, 1]          (lane-shift right by 1, NO bit-rotation)
#             A[x,2] = rotl(A[x,2], 11)   (bit-rotate y=2 lanes by 11)
#   ι      : A[0,0] ^= RC[r]
#   χ      : A[x,y] ^= (~A[x,y+1]) & A[x,y+2]   (mod-3 on y)
#   ρ_east : A[x,1] = rotl(A[x,1], 1)            (bit-rotate y=1 lanes by 1)
#             A[x,2] = rotl(A[x-2, 2], 8)         (lane-shift LEFT by 2, then rotate by 8)
#
# Round constants: last `rounds` entries applied → round=1 is weakest (matches ASCON).
#
# Encrypt interface (reduced-round, consistent with other ciphers here):
#   lanes 0-3  ← key (128 bits)
#   lanes 4-7  ← nonce (128 bits)
#   lanes 8-11 ← 0
#   XOR plaintext into lanes 0-3, permute, squeeze lanes 0-3.
#
# Reference: Daemen et al., "The design of Xoodoo and Xoofff", ToSC 2018(4).
#   Pseudocode: https://keccak.team/xoodoo.html

XOODOO_RC = [
    0x00000058, 0x00000038, 0x000003C0, 0x000000D0,
    0x00000120, 0x00000014, 0x00000060, 0x0000002C,
    0x00000380, 0x000000F0, 0x000001A0, 0x00000012,
]

MASK32 = 0xFFFFFFFF

def xoodoo_permutation(A: list, rounds: int) -> list:
    """
    Xoodoo permutation on 12 32-bit lanes (A[4*y + x] = a[y][x]).
    Applies the LAST `rounds` round constants so round=1 is weakest.
    """
    A = list(A)
    start = 12 - rounds

    for r in range(start, 12):

        # ── θ: column-parity mixing ──────────────────────────────────────
        # P[x] = column parity of x  (XOR over all 3 planes)
        P = [A[x] ^ A[4 + x] ^ A[8 + x] for x in range(4)]
        # E[x] feeds from column (x-1), two rotations
        E = [rotl32(P[(x - 1) & 3], 5) ^ rotl32(P[(x - 1) & 3], 14)
             for x in range(4)]
        # XOR E[x] into every lane in column x (all 3 planes)
        for x in range(4):
            A[x]     ^= E[x]   # y=0
            A[4 + x] ^= E[x]   # y=1
            A[8 + x] ^= E[x]   # y=2

        # ── ρ_west ───────────────────────────────────────────────────────
        # y=1: cyclic lane-shift right by 1 (A[x,1] = A[x-1, 1]), NO bit-rotation
        old_y1 = [A[4 + x] for x in range(4)]
        for x in range(4):
            A[4 + x] = old_y1[(x - 1) & 3]          # lane shift only
        # y=2: bit-rotate each lane left by 11 (no lane shift)
        for x in range(4):
            A[8 + x] = rotl32(A[8 + x], 11)

        # ── ι: round constant into A[0,0] ────────────────────────────────
        A[0] ^= XOODOO_RC[r]

        # ── χ: non-linear, column-wise across y (mod 3) ──────────────────
        # A[x,y] ^= (~A[x,(y+1)%3]) & A[x,(y+2)%3]
        new_A = list(A)
        for x in range(4):
            a0 = A[x];    a1 = A[4 + x];    a2 = A[8 + x]
            new_A[x]     = a0 ^ ((~a1 & MASK32) & a2)
            new_A[4 + x] = a1 ^ ((~a2 & MASK32) & a0)
            new_A[8 + x] = a2 ^ ((~a0 & MASK32) & a1)
        A = new_A

        # ── ρ_east ───────────────────────────────────────────────────────
        # y=1: bit-rotate each lane left by 1 (no lane shift)
        for x in range(4):
            A[4 + x] = rotl32(A[4 + x], 1)
        # y=2: cyclic lane-shift LEFT by 2 (A[x,2] = A[x-2, 2]), then rotate by 8
        old_y2 = [A[8 + x] for x in range(4)]
        for x in range(4):
            A[8 + x] = rotl32(old_y2[(x - 2) & 3], 8)   # shift by 2, then rotate

    return A


def xoodyak_encrypt(P: bytes, rounds: int, key: bytes, nonce: bytes) -> bytes:
    """
    Reduced-round Xoodyak encryption → 128-bit output.
    Little-endian lane packing per Xoodoo/Xoodyak spec.
    """
    def load_lanes(data: bytes, n: int) -> list:
        return [int.from_bytes(data[4*i : 4*i+4], 'little') for i in range(n)]

    A = [0] * 12
    A[0:4]  = load_lanes(key,   4)
    A[4:8]  = load_lanes(nonce, 4)
    A[8:12] = [0, 0, 0, 0]

    # Absorb plaintext into y=0 plane
    for x, lane in enumerate(load_lanes(P, 4)):
        A[x] ^= lane

    A = xoodoo_permutation(A, rounds)

    # Squeeze: lanes 0-3 (y=0 plane) → 16 bytes little-endian
    return b''.join(A[x].to_bytes(4, 'little') for x in range(4))
