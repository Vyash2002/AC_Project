from ciphers.ascon import ascon_encrypt
from ciphers.gift import gift_encrypt
from ciphers.simon import simon_encrypt
from ciphers.speck import speck_encrypt
from ciphers.present import present_encrypt
from ciphers.rectangle import rectangle_encrypt
from ciphers.grain import grain_encrypt
from ciphers.sparx import sparx_encrypt
from ciphers.xoodyak import xoodyak_encrypt
from ciphers.rocca import rocca_encrypt
from ciphers.snow import snowvi_encrypt
from ciphers.skinny import skinny128_encrypt
from ciphers.lea import lea128_encrypt

CIPHER_FN = {
    "ascon": ascon_encrypt,
    "gift": gift_encrypt,
    "simon": simon_encrypt,
    "speck": speck_encrypt,
    "present": present_encrypt,
    "rectangle": rectangle_encrypt,
    "grain": grain_encrypt,
    "sparx": sparx_encrypt,
    "xoodyak": xoodyak_encrypt,
    "rocca": rocca_encrypt,
    "snow" : snowvi_encrypt,
    "skinny": skinny128_encrypt,
    "lea": lea128_encrypt
}


def encrypt(cipher, P, rounds, key, nonce):
    return CIPHER_FN[cipher](P, rounds, key, nonce=nonce)