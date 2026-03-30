from core.dispatch import encrypt
from core.config import BLOCK_SIZE

def run_self_tests(ciphers):
    print("Running self-tests...")

    P = bytes(range(16))
    key = bytes(range(16))
    nonce = bytes(range(16, 32))

    for c in ciphers:
        C1 = encrypt(c, P, 4, key, nonce)
        C2 = encrypt(c, P, 4, key, nonce)

        assert C1 == C2, f"{c} not deterministic"
        assert len(C1) == BLOCK_SIZE, f"{c} wrong output size"

    print("All tests passed ✓")