import hashlib


def simple_shift(text: str) -> str:
    """
    Return the Caesar-shifted string of the input using a fixed key.
    Non-alphabetic characters are left unchanged.
    """
    if not isinstance(text, str):
        raise TypeError("simple_shift expects a string input")
    shift = 3  # fixed key
    result = []
    for ch in text:
        if 'a' <= ch <= 'z':
            base = ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        elif 'A' <= ch <= 'Z':
            base = ord('A')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return ''.join(result)


def _pkcs7_pad(data: bytes, block_size: int) -> bytes:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    pad_len = block_size - (len(data) % block_size)
    if pad_len == 0:
        pad_len = block_size
    return data + bytes([pad_len]) * pad_len


def flexible_encryption(algorithm: str, data: str) -> bytes:
    """
    Encrypt the provided data using the selected algorithm.

    Supported algorithms:
      - 'caesar' (fixed shift=3; returns UTF-8 bytes of shifted text)
      - 'DES' (ECB + PKCS#7 padding; requires pycryptodome)
      - 'AES' (ECB + PKCS#7 padding; requires pycryptodome)
    """
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")
    if not isinstance(data, str):
        raise TypeError("data must be a string")

    alg = algorithm.strip().lower()

    if alg == "caesar":
        return simple_shift(data).encode("utf-8")

    elif alg == "des":
        try:
            from Crypto.Cipher import DES  # type: ignore
        except Exception as e:
            raise RuntimeError("DES encryption requires the 'pycryptodome' package") from e
        key = b"8bytekey"  # 8 bytes for DES
        cipher = DES.new(key, DES.MODE_ECB)
        plaintext = data.encode("utf-8")
        padded = _pkcs7_pad(plaintext, DES.block_size)
        return cipher.encrypt(padded)

    elif alg == "aes":
        try:
            from Crypto.Cipher import AES  # type: ignore
        except Exception as e:
            raise RuntimeError("AES encryption requires the 'pycryptodome' package") from e
        key = b"0123456789abcdef"  # 16 bytes for AES-128
        cipher = AES.new(key, AES.MODE_ECB)
        plaintext = data.encode("utf-8")
        padded = _pkcs7_pad(plaintext, 16)
        return cipher.encrypt(padded)

    else:
        raise ValueError("Unsupported algorithm. Choose from: 'caesar', 'DES', 'AES'.")


def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Apply the specified cryptographic hash to the password and return the hash bytes.
    Raises ValueError if the chosen hash algorithm is not available.
    """
    if not isinstance(algo_name, str):
        raise TypeError("algo_name must be a string")
    if not isinstance(pw, str):
        raise TypeError("pw must be a string")

    name = algo_name.strip().lower()
    data = pw.encode("utf-8")

    # Handle SHAKE algorithms which require an explicit digest length.
    if name.startswith("shake_"):
        try:
            return hashlib.new(name, data).digest(32)  # default to 32-byte output
        except Exception as e:
            raise ValueError(f"Hash algorithm '{algo_name}' is not available") from e

    try:
        return hashlib.new(name, data).digest()
    except Exception as e:
        raise ValueError(f"Hash algorithm '{algo_name}' is not available") from e
