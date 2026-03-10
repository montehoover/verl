import hashlib

UNSAFE_ALGORITHMS = {'md5', 'sha1'}

def compute_digest(algorithm_name: str, content: bytes) -> bytes:
    """
    Computes the hash of byte-like content using the specified algorithm.

    Args:
        algorithm_name: The name of the hash algorithm to use (e.g., 'sha256', 'sha512').
        content: The byte-like object to hash.

    Returns:
        The hash as bytes.

    Raises:
        ValueError: If the specified algorithm is unavailable or unsafe.
    """
    if algorithm_name in UNSAFE_ALGORITHMS:
        raise ValueError(
            f"Hash algorithm '{algorithm_name}' is unsafe and therefore not allowed. "
            f"Unsafe algorithms: {sorted(list(UNSAFE_ALGORITHMS))}"
        )
    if algorithm_name not in hashlib.algorithms_available:
        raise ValueError(
            f"Hash algorithm '{algorithm_name}' is not available. "
            f"Available algorithms: {sorted(list(hashlib.algorithms_available - UNSAFE_ALGORITHMS))}"
        )

    hash_object = hashlib.new(algorithm_name)
    hash_object.update(content)
    return hash_object.digest()

if __name__ == '__main__':
    test_content_string = "hello world"
    test_content_bytes = test_content_string.encode('utf-8')

    # Test with SHA-256
    digest_sha256 = compute_digest('sha256', test_content_bytes)
    print(f"The SHA-256 digest of '{test_content_string}' is: {digest_sha256.hex()}")

    # Test with SHA512
    digest_sha512 = compute_digest('sha512', test_content_bytes)
    print(f"The SHA512 digest of '{test_content_string}' is: {digest_sha512.hex()}")

    # Test with an unsafe algorithm (md5)
    try:
        compute_digest('md5', test_content_bytes)
    except ValueError as e:
        print(f"Error testing md5: {e}")

    # Test with another unsafe algorithm (sha1)
    try:
        compute_digest('sha1', test_content_bytes)
    except ValueError as e:
        print(f"Error testing sha1: {e}")

    # Test with an invalid algorithm
    try:
        compute_digest('invalid_algo', test_content_bytes)
    except ValueError as e:
        print(f"Error testing invalid_algo: {e}")
