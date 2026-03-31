from core.utils import *
# =========================================================
# CIPHER 7 — GRAIN-128AEAD  (Stream cipher, NIST LWC finalist)
# =========================================================
# Grain is a stream cipher: C = keystream XOR P.
# With a standard differential (same key, P vs P^delta), the ciphertext
# difference is always exactly delta regardless of warmup — the keystream
# cancels out.  This makes the distinguishing task trivially solvable at
# every round, producing useless (constant-accuracy) datasets.
#
# Fix: P-derived keying — derived_key = key XOR P.
# Now a 1-bit change in P propagates into the NFSR initial state (key
# register), and the keystream differences grow with warmup clocks.
# This correctly simulates "round complexity" for a stream cipher and
# produces the expected trivial → near-random avalanche curve.
#
# NFSR feedback (full Grain-128AEAD spec):
#   f(b) = s[0] ^ b[0] ^ b[26] ^ b[56] ^ b[91] ^ b[96]
#          ^ b[3]*b[67] ^ b[11]*b[13] ^ b[17]*b[18] ^ b[27]*b[59]
#          ^ b[40]*b[48] ^ b[61]*b[65] ^ b[68]*b[84]
#          ^ b[88]*b[92]*b[93]*b[95] ^ b[22]*b[24]*b[25] ^ b[70]*b[78]*b[82]
#
# Output function h (corrected taps — b[60] is NFSR, not LFSR):
#   h(b,s) = b[12]^s[8] ^ (s[13]&b[20]) ^ (s[42]&b[60]) ^ s[79]
#
# FIX vs original: original had s[60] where spec says b[60].

def _grain_keystream(key: bytes, nonce: bytes, nbits: int,
                     warmup_rounds: int = 256) -> list:
    key_bits   = bytes_to_bits(key)
    nonce_bits = bytes_to_bits(nonce[:12])            # 96-bit nonce

    LFSR = nonce_bits + [1] * 31 + [0]               # 128 bits
    NFSR = key_bits[:]

    def h(b, s):
        x0, x1, x2, x3, x4, x5, x6 = b[12], s[8], s[13], b[20], s[42], b[60], s[79]
        return x0 ^ x1 ^ (x2 & x3) ^ (x4 & x5) ^ x6

    def nfsr_fb(b, s):
        return (
            s[0]  ^ b[0]  ^ b[26] ^ b[56] ^ b[91] ^ b[96]
            ^ (b[3]  & b[67])
            ^ (b[11] & b[13])
            ^ (b[17] & b[18])
            ^ (b[27] & b[59])
            ^ (b[40] & b[48])
            ^ (b[61] & b[65])
            ^ (b[68] & b[84])
            ^ (b[88] & b[92] & b[93] & b[95])
            ^ (b[22] & b[24] & b[25])
            ^ (b[70] & b[78] & b[82])
        )

    def lfsr_fb(s):
        return s[0] ^ s[7] ^ s[38] ^ s[70] ^ s[81] ^ s[96]

    # Warm-up: output y is fed back into both registers
    for _ in range(warmup_rounds):
        y    = h(NFSR, LFSR) ^ LFSR[93] ^ NFSR[2]
        nfb  = nfsr_fb(NFSR, LFSR) ^ y
        lfb  = lfsr_fb(LFSR)        ^ y
        NFSR = NFSR[1:] + [nfb]
        LFSR = LFSR[1:] + [lfb]

    # Keystream generation: y is NOT fed back
    ks = []
    for _ in range(nbits):
        y    = h(NFSR, LFSR) ^ LFSR[93] ^ NFSR[2]
        ks.append(y)
        NFSR = NFSR[1:] + [nfsr_fb(NFSR, LFSR)]
        LFSR = LFSR[1:] + [lfsr_fb(LFSR)]
    return ks

def grain_encrypt(P: bytes, rounds: int, key: bytes, nonce: bytes) -> bytes:
    # P-derived keying: mix plaintext into key so PT diff → key schedule diff
    derived_key = xor_bytes(key, P)
    ks_bits  = _grain_keystream(derived_key, nonce[:12], 128, warmup_rounds=rounds)
    ks_bytes = bytes(
        int("".join(str(b) for b in ks_bits[i:i+8]), 2)
        for i in range(0, 128, 8)
    )
    return xor_bytes(P, ks_bytes)
