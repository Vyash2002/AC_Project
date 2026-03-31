BLOCK_SIZE = 16
NUM_SAMPLES = 100_000
DATA_DIR = "datasets"

DELTA_P      = bytes([0] * 15 + [1])
DELTA_P_BOTH = bytes([0]*7 + [1] + [0]*7 + [1])

CIPHER_DELTA = {
    "ascon": DELTA_P,
    "gift": DELTA_P,
    "simon": DELTA_P,
    "speck": DELTA_P,
    "present": DELTA_P_BOTH,
    "rectangle": DELTA_P_BOTH,
    "grain": DELTA_P,
    "sparx": DELTA_P,
    "xoodyak" : DELTA_P,
    "rocca" : DELTA_P,
    "snow" : DELTA_P,
    "lea" : DELTA_P,
    "skinny" : DELTA_P
}

CIPHER_ROUNDS = {
    "ascon":     [1,2,3,4,5,6,7,8],
    "gift":      [1,2,3,4,5,6,7,8],
    "simon":     [4,8,12,16,20,24,28,32],
    "speck":     [1,2,3,4,5,6,7,8],
    "present":   [1,2,3,4,5,6,7,8],
    "rectangle": [1,2,3,4,5,6,7,8],
    "grain":     [1,2,4,8,16,32,64,128,156,192,256],
    "sparx":     [1,2,3,4,5,6,7,8],
    "xoodyak":   [1, 2, 3, 4, 5, 6, 7, 8],
    "rocca" :    [1,2,3,4,5,6,7,8],
    "snow"  :    [1,2,3,4,5,6,7,8],
    "skinny" :   [1,2,3,4,5,6,7,8],
    "lea" :      [2,4,6,8,10,12,14,16]
}