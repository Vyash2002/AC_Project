import os
import argparse

from core.config import *
from data.dataset import generate_dataset
from tests.selftest import run_self_tests

ALL_CIPHERS = list(CIPHER_ROUNDS.keys())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cipher", nargs="+", choices=ALL_CIPHERS)
    p.add_argument("--rounds", nargs="+", type=int)
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--no-test", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    ciphers = args.cipher if args.cipher else ALL_CIPHERS

    if not args.no_test:
        run_self_tests(ciphers)

    os.makedirs(args.data_dir, exist_ok=True)

    for cipher in ciphers:
        rounds = args.rounds if args.rounds else CIPHER_ROUNDS[cipher]

        for r in rounds:
            print(f"{cipher} r={r}")
            generate_dataset(cipher, r, args.num_samples, args.data_dir)

if __name__ == "__main__":
    main()


'''

This is the main file to generate datasets from ciphers


To generate datasets for all ciphers:
python generate_datasets.py


To generate dataset for specific cipher (for eg : gift)
python generate_datasets.py --cipher gift


To generate dataset for specific cipher and specific number of rounds (for eg : gift , rounds = 2,4,6,8)
python generate_datasets.py --cipher gift --rounds 2 4 6 8


To generate dataset for x samples
python generate_datasets.py --cipher gift --num-samples 20000

'''