import hashlib

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
