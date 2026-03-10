import hashlib

# Define a set of secure algorithms.
# Users should choose based on their security requirements.
# Excluding older/weaker algorithms like MD5, SHA1 for password hashing.
ALLOWED_SECURE_ALGORITHMS = {
    'sha256', 'sha384', 'sha512',
    'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s',
    'sha512_224', 'sha512_256' # These are truncated versions of SHA512
}

def secure_hash(algo_name: str, pw: str) -> bytes:
    """
    Hashes a password using a specified secure cryptographic algorithm.

    Args:
        algo_name: The name of the cryptographic algorithm to be used.
                   Must be one of the ALLOWED_SECURE_ALGORITHMS.
        pw: The password string to hash.

    Returns:
        The resulting hash value as bytes.

    Raises:
        ValueError: If the chosen hash algorithm is not in the allowed list
                    or not available in hashlib.
    """
    if algo_name not in ALLOWED_SECURE_ALGORITHMS:
        raise ValueError(
            f"Algorithm '{algo_name}' is not in the list of allowed secure algorithms. "
            f"Choose from: {', '.join(sorted(list(ALLOWED_SECURE_ALGORITHMS)))}."
        )

    if algo_name not in hashlib.algorithms_available:
        # This check is somewhat redundant if ALLOWED_SECURE_ALGORITHMS
        # is curated from hashlib.algorithms_available, but good for robustness.
        raise ValueError(
            f"Algorithm '{algo_name}' is not available in the current hashlib environment."
        )

    try:
        # Encode the password string to bytes (UTF-8 is a common encoding)
        password_bytes = pw.encode('utf-8')

        # Create a new hash object
        hasher = hashlib.new(algo_name)

        # Update the hasher with the password bytes
        hasher.update(password_bytes)

        # Return the digest
        return hasher.digest()
    except Exception as e:
        # Catch any unexpected errors during hashing, though hashlib.new and update are quite robust.
        raise ValueError(f"Error during hashing with {algo_name}: {e}")

if __name__ == '__main__':
    # Example Usage:
    try:
        # Test with a valid algorithm and password
        hashed_pw = secure_hash("sha256", "password123")
        print(f"SHA256 hash of 'password123': {hashed_pw.hex()}")

        # Expected output for sha256 and "password123" (encoded as b"password123")
        expected_hash = hashlib.sha256(b"password123").digest()
        print(f"Expected SHA256 hash: {expected_hash.hex()}")
        assert hashed_pw == expected_hash

        hashed_pw_blake2b = secure_hash("blake2b", "another-secure-password!")
        print(f"BLAKE2b hash of 'another-secure-password!': {hashed_pw_blake2b.hex()}")

        # Test with an algorithm not in our allowed list
        try:
            secure_hash("md5", "test")
        except ValueError as e:
            print(f"Caught expected error for md5: {e}")

        # Test with an algorithm in allowed list but hypothetically not in hashlib.algorithms_available
        # (This requires manually removing an algo from ALLOWED_SECURE_ALGORITHMS if it's always available)
        # For demonstration, let's assume 'non_existent_algo' was in ALLOWED_SECURE_ALGORITHMS
        # but not in hashlib.algorithms_available
        # To actually test this, you'd need to manipulate hashlib or use a truly non-existent name.
        # If 'sha256' is removed from hashlib.algorithms_available for some reason:
        # original_algorithms_available = hashlib.algorithms_available
        # hashlib.algorithms_available = {'sha512'} # Simulate sha256 not being available
        # try:
        #     secure_hash("sha256", "test")
        # except ValueError as e:
        #     print(f"Caught expected error for unavailable algo: {e}")
        # hashlib.algorithms_available = original_algorithms_available # Restore

    except ValueError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
