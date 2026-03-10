from Crypto.Cipher import AES, DES
from Crypto.Util.Padding import pad

# IMPORTANT: In a real application, key management is critical and hardcoding keys is insecure.
# These keys are fixed for demonstration purposes based on the problem's context.
_FIXED_DES_KEY = b'8byteDESkey!'  # DES key must be 8 bytes
_FIXED_AES_KEY = b'16byteAESkey!!!!'  # AES key can be 16, 24, or 32 bytes. This is 16 bytes for AES-128.

def _caesar_cipher(text: str, key: int = 3) -> bytes:
    """
    Applies a Caesar cipher shift to the input string.

    Args:
        text: The input string to be shifted.
        key: The shift key for the Caesar cipher.

    Returns:
        The Caesar shifted string, encoded as bytes.
    """
    shifted_chars = []
    for char_code in [ord(c) for c in text]:
        # Uppercase letters
        if 65 <= char_code <= 90:  # ord('A') to ord('Z')
            shifted_char_code = (char_code - 65 + key) % 26 + 65
        # Lowercase letters
        elif 97 <= char_code <= 122:  # ord('a') to ord('z')
            shifted_char_code = (char_code - 97 + key) % 26 + 97
        # Other characters
        else:
            shifted_char_code = char_code
        shifted_chars.append(chr(shifted_char_code))
    return "".join(shifted_chars).encode('utf-8')

def flexible_encryption(algorithm: str, data: str) -> bytes:
    """
    Encrypts data using the specified algorithm.

    Args:
        algorithm: The encryption algorithm to use ('caesar', 'DES', 'AES').
        data: The string data to encrypt.

    Returns:
        The encrypted data as bytes. For 'caesar', it's the shifted text encoded to UTF-8.
        For DES and AES (using CBC mode), the IV is prepended to the ciphertext.

    Raises:
        ValueError: If an unsupported algorithm is specified.
        ImportError: If pycryptodome is not installed (for DES/AES).
    """
    data_bytes = data.encode('utf-8') # Ensure data is bytes for encryption where needed

    if algorithm.lower() == 'caesar':
        # Using a fixed key of 3, similar to the original simple_shift
        return _caesar_cipher(data, key=3)
    elif algorithm.lower() == 'des':
        # Using CBC mode for DES. DES.new() generates a random IV for CBC if not provided.
        cipher = DES.new(_FIXED_DES_KEY, DES.MODE_CBC)
        iv = cipher.iv  # The randomly generated IV
        padded_data = pad(data_bytes, DES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        # Prepend IV to the ciphertext for use during decryption
        return iv + encrypted_data
    elif algorithm.lower() == 'aes':
        # Using CBC mode for AES. AES.new() generates a random IV for CBC if not provided.
        cipher = AES.new(_FIXED_AES_KEY, AES.MODE_CBC)
        iv = cipher.iv  # The randomly generated IV
        padded_data = pad(data_bytes, AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        # Prepend IV to the ciphertext for use during decryption
        return iv + encrypted_data
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms are 'caesar', 'DES', 'AES'.")
