# =========================================================
# CIPHER 5 — PRESENT-128  (SPN, Bogdanov et al. 2007)
# =========================================================
# 64-bit block, 128-bit key.  We encrypt both 8-byte halves of P
# independently (with domain-separated keys) to produce 128-bit output.
_PRESENT_SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
                 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
_PRESENT_PERM = [16 * (i % 4) + (i // 4) for i in range(64)]

def present_encrypt(P: bytes, rounds: int, key: bytes, **_) -> bytes:
    def _half(block: int, rounds: int, k: int) -> int:
        state = block & 0xFFFF_FFFF_FFFF_FFFF
        k    &= (1 << 128) - 1
        subkeys = []
        for r in range(rounds + 1):
            subkeys.append((k >> 64) & 0xFFFF_FFFF_FFFF_FFFF)
            k = ((k << 61) | (k >> 67)) & ((1 << 128) - 1)
            top = (k >> 124) & 0xF
            k   = (k & ~(0xF << 124)) | (_PRESENT_SBOX[top] << 124)
            k  ^= ((r + 1) & 0x1F) << 62
        for r in range(rounds):
            state ^= subkeys[r]
            ns = 0
            for nib in range(16):
                sh = (15 - nib) * 4
                ns |= _PRESENT_SBOX[(state >> sh) & 0xF] << sh
            state = ns
            if r < rounds - 1:
                ns = 0
                for i in range(64):
                    ns |= ((state >> (63 - i)) & 1) << (63 - _PRESENT_PERM[i])
                state = ns
        state ^= subkeys[rounds]
        return state

    k_int  = int.from_bytes(key, 'big')
    lo     = int.from_bytes(P[:8], 'big')
    hi     = int.from_bytes(P[8:], 'big')
    out_lo = _half(lo, rounds, k_int)
    out_hi = _half(hi, rounds, k_int ^ 0xDEAD_BEEF_DEAD_BEEF)
    return out_lo.to_bytes(8, 'big') + out_hi.to_bytes(8, 'big')

