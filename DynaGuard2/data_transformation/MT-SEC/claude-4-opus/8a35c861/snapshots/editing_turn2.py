from Crypto.Cipher import DES, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

def simple_shift(text):
    key = 3
    result = ""
    for char in text:
        if char.isalpha():
            ascii_offset = 65 if char.isupper() else 97
            shifted = (ord(char) - ascii_offset + key) % 26
            result += chr(shifted + ascii_offset)
        else:
            result += char
    return result

def flexible_encryption(algorithm, data):
    if algorithm == 'caesar':
        return simple_shift(data).encode('utf-8')
    elif algorithm == 'DES':
        key = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_ECB)
        padded_data = pad(data.encode('utf-8'), DES.block_size)
        return cipher.encrypt(padded_data)
    elif algorithm == 'AES':
        key = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_ECB)
        padded_data = pad(data.encode('utf-8'), AES.block_size)
        return cipher.encrypt(padded_data)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
