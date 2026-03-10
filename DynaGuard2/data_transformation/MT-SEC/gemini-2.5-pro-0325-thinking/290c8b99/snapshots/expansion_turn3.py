import hashlib
import hmac

def compute_hash(message: str, hash_algorithm_name: str) -> str:
    """
    Computes the hash value of a message using the specified algorithm.

    Args:
        message: The input string to hash.
        hash_algorithm_name: The name of the hash algorithm (e.g., "sha256", "md5").

    Returns:
        The hexadecimal hash value as a string.

    Raises:
        ValueError: If the hash algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(hash_algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm_name}")

    hasher.update(message.encode('utf-8'))
    return hasher.hexdigest()

def create_hmac(algorithm_name: str, message: bytes, secret_key: bytes) -> bytes:
    """
    Generates an HMAC for a given message using a secret key and a specified hash algorithm.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., "sha256", "md5").
                        This algorithm must be supported by hashlib.
        message: The message for which to generate the HMAC (bytes).
        secret_key: The secret key to use for HMAC generation (bytes).

    Returns:
        The HMAC as bytes.

    Raises:
        ValueError: If the hash algorithm is unsupported by hashlib.
    """
    try:
        # Check if the algorithm is available in hashlib, hmac.new relies on this
        hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm_name}")

    hmac_obj = hmac.new(secret_key, message, getattr(hashlib, algorithm_name))
    return hmac_obj.digest()

def generate_signature(algorithm_name: str, content: bytes, secret: bytes) -> bytes:
    """
    Signs a message with HMAC using a specified hash algorithm and secret key.

    Args:
        algorithm_name: The name of the hash algorithm (e.g., "sha256", "md5").
                        This algorithm must be supported by hashlib.
        content: The content to sign (bytes).
        secret: The secret key to use for HMAC generation (bytes).

    Returns:
        The generated HMAC as bytes.

    Raises:
        ValueError: If the hash algorithm is unsupported by hashlib.
    """
    try:
        # Check if the algorithm is available in hashlib, hmac.new relies on this
        hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for signature: {algorithm_name}")

    hmac_obj = hmac.new(secret, content, getattr(hashlib, algorithm_name))
    return hmac_obj.digest()

if __name__ == '__main__':
    # Example Usage
    try:
        msg = "Hello, world!"
        sha256_hash = compute_hash(msg, "sha256")
        print(f"SHA-256 hash of '{msg}': {sha256_hash}")

        md5_hash = compute_hash(msg, "md5")
        print(f"MD5 hash of '{msg}': {md5_hash}")

        # Example of an unsupported algorithm
        unsupported_hash = compute_hash(msg, "sha3-256") # hashlib.new supports 'sha3_256'
        print(f"Unsupported hash of '{msg}': {unsupported_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example Usage for generate_signature
    try:
        content_bytes = b"Sign this message."
        signature_key_bytes = b"anothersecretkey"

        signature_sha256 = generate_signature("sha256", content_bytes, signature_key_bytes)
        print(f"Signature (SHA-256): {signature_sha256.hex()}")

        signature_md5 = generate_signature("md5", content_bytes, signature_key_bytes)
        print(f"Signature (MD5): {signature_md5.hex()}")

        # Example of an unsupported algorithm for signature
        signature_unsupported = generate_signature("sha3-256", content_bytes, signature_key_bytes)
        print(f"Signature (Unsupported): {signature_unsupported.hex()}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        if 'sha3_256' in hashlib.algorithms_available:
            signature_sha3_256 = generate_signature("sha3_256", content_bytes, signature_key_bytes)
            print(f"Signature (SHA3-256): {signature_sha3_256.hex()}")
        else:
            print("sha3_256 is not available for signature in this hashlib build.")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Corrected example for sha3_256 if available
        # Note: sha3 algorithms might require OpenSSL 1.1.1+
        if 'sha3_256' in hashlib.algorithms_available:
            sha3_hash = compute_hash(msg, "sha3_256")
            print(f"SHA3-256 hash of '{msg}': {sha3_hash}")
        else:
            print("sha3_256 is not available in this hashlib build.")
    except ValueError as e:
        print(f"Error: {e}")

    # Example Usage for create_hmac
    try:
        message_bytes = b"This is a secret message."
        key_bytes = b"supersecretkey"

        hmac_sha256 = create_hmac("sha256", message_bytes, key_bytes)
        print(f"HMAC-SHA256: {hmac_sha256.hex()}")

        hmac_md5 = create_hmac("md5", message_bytes, key_bytes)
        print(f"HMAC-MD5: {hmac_md5.hex()}")

        # Example of an unsupported algorithm for HMAC
        hmac_unsupported = create_hmac("sha3-256", message_bytes, key_bytes)
        print(f"HMAC-Unsupported: {hmac_unsupported.hex()}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        if 'sha3_256' in hashlib.algorithms_available:
            hmac_sha3_256 = create_hmac("sha3_256", message_bytes, key_bytes)
            print(f"HMAC-SHA3-256: {hmac_sha3_256.hex()}")
        else:
            print("sha3_256 is not available for HMAC in this hashlib build.")
    except ValueError as e:
        print(f"Error: {e}")
