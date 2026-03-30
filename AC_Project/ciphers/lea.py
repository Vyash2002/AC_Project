from core.utils import *

# =========================================================
# CIPHER 13 — LEA-128  (Lim & Koo, WISA 2013; ISO/IEC 29192-2)
# =========================================================
# Pure ARX block cipher — no S-boxes whatsoever.
# Block : 128 bits  = four 32-bit words (X0, X1, X2, X3)
# Key   : 128 bits  = four 32-bit words (K0, K1, K2, K3)
# Full cipher uses 32 rounds; this implementation exposes `rounds`
# as a parameter so reduced-round variants can be studied.
#
# Round function (identical structure for every round r):
#
#   t0 = rotl32((X0 XOR RK[r][0]) + (X1 XOR RK[r][1]),  9)
#   t1 = rotr32((X1 XOR RK[r][2]) + (X2 XOR RK[r][3]),  5)
#   t2 = rotr32((X2 XOR RK[r][4]) + (X3 XOR RK[r][5]),  3)
#   (X0, X1, X2, X3) <- (t0, t1, t2, X0)      # X3 rotates out
#
# Note: the spec describes even/odd rounds with different rotation
# directions for some operations, but the rotation amounts and the
# feedforward of X0 into X3 are identical for all rounds, so a single
# branch is sufficient here.
#
# Key schedule:
#   Input : K = [K0, K1, K2, K3] in little-endian 32-bit words.
#   Working copy T[0..3] is updated each round using 8 cyclic constants δ.
#   Round i produces 6 subkey words:
#     T[0] = rotl32((T[0] + rotl32(δ[i%8],  i   )) mod 2^32,  1)
#     T[1] = rotl32((T[1] + rotl32(δ[i%8],  i+1 )) mod 2^32,  3)
#     T[2] = rotl32((T[2] + rotl32(δ[i%8],  i+2 )) mod 2^32,  6)
#     T[3] = rotl32((T[3] + rotl32(δ[i%8],  i+3 )) mod 2^32, 11)
#     RK[i] = (T[0], T[1], T[2], T[1], T[3], T[1])
#
# Constants δ[0..7] (Table 2 of the LEA specification):
#   0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec,
#   0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957
#
# State and key words are stored in LITTLE-ENDIAN byte order,
# consistent with the LEA reference implementation.
#
# Reference: Lim & Koo, "A Compact and Fast VLSI Architecture for
#   LEA Block Cipher", WISA 2013; ISO/IEC 29192-2:2019.
# =========================================================

# Eight cyclic delta constants for the key schedule (Table 2, LEA spec)
_LEA_DELTA = [
    0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec,
    0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957,
]


def _lea_key_schedule(key: bytes, rounds: int) -> list:
    """
    Derive the LEA-128 round-key table from a 128-bit key.

    Parameters
    ----------
    key    : 16-byte key (interpreted as four little-endian 32-bit words)
    rounds : number of rounds to generate subkeys for

    Returns
    -------
    RK : list of `rounds` tuples, each containing 6 x 32-bit subkey words.
         RK[i] = (T0, T1, T2, T1, T3, T1)  — T1 appears three times
                 as specified in the LEA key schedule.
    """
    # Load key as four little-endian 32-bit words
    K = [int.from_bytes(key[i:i+4], 'little') for i in range(0, 16, 4)]
    T = list(K)    # T[0..3]: working copy updated each round
    RK = []

    for i in range(rounds):
        delta = _LEA_DELTA[i % 8]

        # Each T word is updated by adding a rotated delta, then rotating the result
        T[0] = rotl32((T[0] + rotl32(delta, i    )) & MASK32,  1)
        T[1] = rotl32((T[1] + rotl32(delta, i + 1)) & MASK32,  3)
        T[2] = rotl32((T[2] + rotl32(delta, i + 2)) & MASK32,  6)
        T[3] = rotl32((T[3] + rotl32(delta, i + 3)) & MASK32, 11)

        # T[1] appears in positions 1, 3, and 5 of every round-key tuple
        RK.append((T[0], T[1], T[2], T[1], T[3], T[1]))

    return RK


def lea128_encrypt(P: bytes, rounds: int, key: bytes, **_) -> bytes:
    """
    LEA-128 reduced-round encryption.

    Parameters
    ----------
    P      : 16-byte plaintext (interpreted as four little-endian 32-bit words)
    rounds : number of rounds to apply (1 ≤ rounds ≤ 32)
    key    : 16-byte key

    Returns
    -------
    16-byte ciphertext (four little-endian 32-bit words, concatenated)

    Round structure
    ---------------
    Each round r applies the following to state (X0, X1, X2, X3):

        t0 = rotl32((X0 ^ RK[r][0]) + (X1 ^ RK[r][1]) mod 2^32,  9)
        t1 = rotr32((X1 ^ RK[r][2]) + (X2 ^ RK[r][3]) mod 2^32,  5)
        t2 = rotr32((X2 ^ RK[r][4]) + (X3 ^ RK[r][5]) mod 2^32,  3)
        (X0, X1, X2, X3) <- (t0, t1, t2, X0)

    The modular addition is the sole source of non-linearity.
    No S-box is used anywhere in LEA.

    Note on operator precedence
    ---------------------------
    The additions (X ^ RK[i]) + (Y ^ RK[j]) are computed as:
        ((X ^ RK[i]) + (Y ^ RK[j])) & MASK32
    The & MASK32 is applied BEFORE the rotation to keep all arithmetic
    in the 32-bit domain. Python's arbitrary-precision integers would
    otherwise silently produce correct results even without the mask,
    but the explicit mask documents the intended word width.
    """
    # Load plaintext as four little-endian 32-bit words
    X = [int.from_bytes(P[i:i+4], 'little') for i in range(0, 16, 4)]

    RK = _lea_key_schedule(key, rounds)

    for i in range(rounds):
        rk = RK[i]

        # Compute the three new state words
        t0 = rotl32(((X[0] ^ rk[0]) + (X[1] ^ rk[1])) & MASK32, 9)
        t1 = rotr32(((X[1] ^ rk[2]) + (X[2] ^ rk[3])) & MASK32, 5)
        t2 = rotr32(((X[2] ^ rk[4]) + (X[3] ^ rk[5])) & MASK32, 3)

        # Feedforward: old X0 becomes new X3
        X = [t0, t1, t2, X[0]]

    # Serialise back to bytes (little-endian per word)
    return b''.join(x.to_bytes(4, 'little') for x in X)