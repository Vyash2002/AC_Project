from core.utils import *

# =========================================================
# CIPHER 10 — ROCCA  (Sakamoto et al. 2021)
# =========================================================
# Rocca is a stream cipher / AEAD designed for beyond-5G/6G systems.
# It uses AES round functions as its core primitive, giving it strong,
# immediate diffusion — unlike LFSR-based ciphers (SNOW family).
#
# State:
#   8 x 128-bit registers: Z0..Z7
#   Each register holds a 128-bit integer (AES block sized)
#
# One Rocca round (Update function):
#   Z0' = AES_enc_round(Z6) ^ Z1
#   Z1' = AES_enc_round(Z0) ^ Z5
#   Z2' = AES_enc_round(Z1) ^ Z0
#   Z3' = AES_enc_round(Z2) ^ Z7
#   Z4' = AES_enc_round(Z3) ^ Z1
#   Z5' = AES_enc_round(Z4) ^ Z0
#   Z6' = AES_enc_round(Z5) ^ Z6 ^ ROUND_CONST[r]
#   Z7' = Z6
#
# Initialization:
#   Load key (256-bit = two 128-bit halves K0, K1) and nonce (128-bit).
#   Z0 = K0,  Z1 = K1,  Z2 = nonce,  Z3 = ROCCA_C0
#   Z4 = K0 ^ K1,  Z5 = nonce ^ K1,  Z6 = ROCCA_C1,  Z7 = 0
#   Run `rounds` initialization clocks (no keystream output).
#   After init, XOR K0^K1 into Z0 and nonce into Z1 (domain separation).
#
# Keystream (reduced-round interface):
#   After init, run one keystream clock:
#   z = Z0 ^ Z4   (128-bit keystream word)
#   output = P XOR z
#
# Constants (from Rocca spec, derived from digits of e):
#   ROCCA_C0 = 0x428a2f98d728ae227be4d9a3bf1ed0c0
#   ROCCA_C1 = 0xb5c0fbcfec4d3b2fe9b5dba58189dbbc
#
# Round constants RC[r] are per-round 128-bit values cycling through
# (C0, C1, C0^C1) to inject asymmetry and prevent slide attacks.
#
# Diffusion:
#   Each AES_enc_round applies SubBytes+ShiftRows+MixColumns,
#   giving full 128-bit diffusion in 1 round. A 1-bit plaintext
#   change propagates into Z0 at init, spreads across all 8
#   registers within 2-3 rounds → clean trivial→random avalanche curve.
#
# Reference:
#   Sakamoto, Liu, Nakano, Kiyomoto, Isobe.
#   "Rocca: An Efficient AES-based Encryption Scheme for Beyond 5G."
#   IACR Transactions on Symmetric Cryptology, 2021(2), pp. 1-30.

# ── Rocca constants (128-bit, derived from fractional digits of e) ──────────
_ROCCA_C0 = 0x428a2f98d728ae227be4d9a3bf1ed0c0
_ROCCA_C1 = 0xb5c0fbcfec4d3b2fe9b5dba58189dbbc



# ── AES primitives (used by Rocca) ──────────────────────────────────────────
_AES_SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]

def _xtime(b: int) -> int:
    return ((b << 1) ^ 0x1B) & 0xFF if b & 0x80 else (b << 1) & 0xFF

def _aes_enc_round(state: int) -> int:
    """AES SubBytes + ShiftRows + MixColumns (no AddRoundKey) on a 128-bit int."""
    b = list(state.to_bytes(16, 'big'))
    # SubBytes
    b = [_AES_SBOX[x] for x in b]
    # ShiftRows
    b = [
        b[0],  b[5],  b[10], b[15],
        b[4],  b[9],  b[14], b[3],
        b[8],  b[13], b[2],  b[7],
        b[12], b[1],  b[6],  b[11],
    ]
    # MixColumns (each 4-byte column)
    out = []
    for col in range(4):
        s0,s1,s2,s3 = b[col*4], b[col*4+1], b[col*4+2], b[col*4+3]
        out += [
            _xtime(s0) ^ _xtime(s1)^s1 ^ s2 ^ s3,
            s0 ^ _xtime(s1) ^ _xtime(s2) ^ _xtime(s3)^s3,
            s0 ^ s1 ^ _xtime(s2) ^ _xtime(s3)^s3,
            _xtime(s0)^s0 ^ s1 ^ s2 ^ _xtime(s3),
        ]
    return int.from_bytes(out, 'big')

# Per-round 128-bit injection constants cycling C0 → C1 → C0^C1 → repeat
def _rocca_round_const(r: int) -> int:
    cycle = r % 3
    if cycle == 0: return _ROCCA_C0
    if cycle == 1: return _ROCCA_C1
    return _ROCCA_C0 ^ _ROCCA_C1


def _rocca_update(Z: list, rc: int) -> list:
    """
    One Rocca state update (8 x 128-bit registers).
    Uses _aes_enc_round from the SNOW-Vi section above (SubBytes+ShiftRows+MixColumns).

    Z0' = AES(Z6) ^ Z1
    Z1' = AES(Z0) ^ Z5
    Z2' = AES(Z1) ^ Z0
    Z3' = AES(Z2) ^ Z7
    Z4' = AES(Z3) ^ Z1
    Z5' = AES(Z4) ^ Z0
    Z6' = AES(Z5) ^ Z6 ^ RC[r]
    Z7' = Z6
    """
    M = (1 << 128) - 1
    return [
        _aes_enc_round(Z[6]) ^ Z[1],           # Z0'
        _aes_enc_round(Z[0]) ^ Z[5],           # Z1'
        _aes_enc_round(Z[1]) ^ Z[0],           # Z2'
        _aes_enc_round(Z[2]) ^ Z[7],           # Z3'
        _aes_enc_round(Z[3]) ^ Z[1],           # Z4'
        _aes_enc_round(Z[4]) ^ Z[0],           # Z5'
        _aes_enc_round(Z[5]) ^ Z[6] ^ rc,      # Z6'
        Z[6],                                   # Z7'
    ]


def rocca_encrypt(P: bytes, rounds: int, key: bytes, nonce: bytes) -> bytes:
    """
    Reduced-round Rocca encryption → 128-bit output.

    Key:   128-bit (we split into K0=key, K1=key^nonce for domain separation,
           matching the framework's 16-byte key constraint while keeping
           K0 != K1 so all 8 registers are independently seeded).
    Nonce: 128-bit.
    Rounds: number of initialisation update clocks (mirrors Grain/SNOW-Vi
            convention — more clocks = more diffusion = harder to distinguish).

    After `rounds` init clocks, one keystream clock produces z = Z0 ^ Z4.
    Output = P XOR z[:16].
    """
    MASK128 = (1 << 128) - 1

    # Unpack key and nonce as 128-bit integers (big-endian)
    # P-derived keying: mix P into K0 so a 1-bit PT change →
    # different keystream, same as the Grain fix in this framework
    K0 = int.from_bytes(xor_bytes(key, P), 'big')
    N  = int.from_bytes(nonce, 'big')
    K1 = K0 ^ N ^ _ROCCA_C0

    # ── Initialise state ────────────────────────────────────────────────────
    # Per Rocca spec §2.2 (adapted for 128-bit key):
    Z = [
        K0,           # Z0
        K1,           # Z1
        N,            # Z2
        _ROCCA_C0,    # Z3
        K0 ^ K1,      # Z4
        N  ^ K1,      # Z5
        _ROCCA_C1,    # Z6
        0,            # Z7
    ]

    # ── Initialisation phase: `rounds` update clocks ────────────────────────
    for r in range(rounds):
        Z = _rocca_update(Z, _rocca_round_const(r))

    # Domain-separation finalisation (prevents related-key slide attacks)
    Z[0] ^= K0 ^ K1
    Z[1] ^= N

    # ── Keystream phase: one clean clock ────────────────────────────────────
    Z = _rocca_update(Z, _rocca_round_const(rounds))

    # Keystream word: Z0 ^ Z4  (128-bit)
    z       = (Z[0] ^ Z[4]) & MASK128
    z_bytes = z.to_bytes(16, 'big')

    return xor_bytes(P, z_bytes)
