from core.utils import *

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

#Cipher 11
# ── SNOW-Vi GF(2^32) arithmetic ─────────────────────────────────────────────
# Primitive polynomial: p(x) = x^32 + x^22 + x^2 + x + 1  → lower 32 bits = 0x00400007
_SVI_POLY     = 0x00400007
_SVI_POLY_INV = 0x80200003   # (0x00400007 >> 1) | 0x80000000  for α^-1

def _svi_mulalpha(v: int) -> int:
    return ((v << 1) ^ (_SVI_POLY if (v >> 31) & 1 else 0)) & 0xFFFFFFFF

def _svi_mulalpha_inv(v: int) -> int:
    return ((v >> 1) ^ (_SVI_POLY_INV if v & 1 else 0)) & 0xFFFFFFFF

def _svi_lfsr_step(s: list) -> list:
    """Advance LFSR one 32-bit step.
       Recurrence: s[t+16] = s[t+15] ^ mulα(s[t+11]) ^ s[t+2] ^ mulα_inv(s[t])
    """
    new_s = s[15] ^ _svi_mulalpha(s[11]) ^ s[2] ^ _svi_mulalpha_inv(s[0])
    return s[1:] + [new_s]

def _svi_pack128(s: list, i: int) -> int:
    """Pack 4 consecutive 32-bit LFSR words starting at index i → 128-bit int."""
    return (s[i] << 96) | (s[i+1] << 64) | (s[i+2] << 32) | s[i+3]

def snowvi_encrypt(P: bytes, rounds: int, key: bytes, nonce: bytes) -> bytes:
    """
    Reduced-round SNOW-Vi encryption → 128-bit output.

    Structure:
      - LFSR : 16 × 32-bit words over GF(2^32)
      - FSM  : 4 × 128-bit registers (R1..R4) using _aes_enc_round
      - Output: 128-bit keystream per clock  (z = F ^ S_lower)

    P-derived keying (critical fix — same bug as Grain/Rocca):
      Without it: C1 ^ C2 = delta always, CNN trivially accurate.
      Fix: dk = key ^ P fed into LFSR init so 1-bit PT diff →
      different LFSR seed → different keystream → real avalanche.
    """
    MASK128 = (1 << 128) - 1

    # ── P-derived keying ────────────────────────────────────────────────────
    dk = xor_bytes(key, P)

    # Unpack derived key and nonce as 4 × 32-bit words (big-endian)
    k = [int.from_bytes(dk[i:i+4],    'big') for i in range(0, 16, 4)]
    n = [int.from_bytes(nonce[i:i+4], 'big') for i in range(0, 16, 4)]

    # ── LFSR init ───────────────────────────────────────────────────────────
    # s[0..3]   = key words  (oldest)
    # s[4..7]   = key words ^ 0xFFFFFFFF  (domain-separated copy)
    # s[8..11]  = nonce words
    # s[12..15] = nonce ^ key words  (newest, further mixing)
    s = (
        k[:]                              +   # s[0..3]
        [w ^ 0xFFFFFFFF for w in k]      +   # s[4..7]
        n[:]                              +   # s[8..11]
        [nw ^ kw for nw, kw in zip(n, k)]    # s[12..15]
    )

    # ── FSM init from packed LFSR halves ────────────────────────────────────
    R1 = _svi_pack128(s,  0)
    R2 = _svi_pack128(s,  4)
    R3 = _svi_pack128(s,  8)
    R4 = _svi_pack128(s, 12)

    # ── Initialization phase: `rounds` FSM clocks ───────────────────────────
    # Each clock = 4 LFSR steps (produces 128 bits) + FSM update.
    # Keystream z is fed back into LFSR (not output) during init.
    for _ in range(rounds):
        S_upper = _svi_pack128(s, 12)   # newest 128 bits of LFSR
        S_lower = _svi_pack128(s,  0)   # oldest 128 bits of LFSR

        # FSM update  (AES round gives full 128-bit diffusion per step)
        F      = (_aes_enc_round(R1 ^ S_upper) ^ R2) & MASK128
        R1_new = F
        R2_new = _aes_enc_round(R1)
        R3_new = _aes_enc_round(R2)
        R4_new = _aes_enc_round(R3)

        # Keystream word (fed back into LFSR, not output yet)
        z       = (F ^ S_lower) & MASK128
        z_words = [(z >> (96 - 32 * i)) & 0xFFFFFFFF for i in range(4)]

        # Advance LFSR 4 × 32-bit steps; inject z feedback during init
        for j in range(4):
            s[-1] ^= z_words[j]     # keystream feedback into newest LFSR cell
            s = _svi_lfsr_step(s)

        R1, R2, R3, R4 = R1_new, R2_new, R3_new, R4_new

    # ── Keystream phase: one clean clock (no feedback) ──────────────────────
    S_upper = _svi_pack128(s, 12)
    S_lower = _svi_pack128(s,  0)

    F       = (_aes_enc_round(R1 ^ S_upper) ^ R2) & MASK128
    z       = (F ^ S_lower) & MASK128
    z_bytes = z.to_bytes(16, 'big')

    return xor_bytes(P, z_bytes)
