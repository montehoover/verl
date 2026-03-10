import hashlib
import hmac

secret_key = b'supersecret'

def hash_data(data: str, algorithm_name: str) -> str:
    """
    Hashes the input data using the specified algorithm.

    Args:
        data: The string data to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash value of the input data.

    Raises:
        ValueError: If the specified algorithm is unsupported.
    """
    try:
        hasher = hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}")
    
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()

def generate_hmac(message: bytes, algorithm_name: str) -> str:
    """
    Generates an HMAC for the given message using the specified hash algorithm.

    Args:
        message: The message to authenticate (bytes).
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').
                        This algorithm must be supported by hashlib.

    Returns:
        The hexadecimal HMAC value.

    Raises:
        ValueError: If the specified algorithm is unsupported by hashlib.
    """
    try:
        # Check if algorithm is supported by hashlib, hmac.new relies on this
        hashlib.new(algorithm_name)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm for HMAC: {algorithm_name}")

    hmac_generator = hmac.new(secret_key, message, getattr(hashlib, algorithm_name))
    return hmac_generator.hexdigest()

def sign_data(algorithm: str, message: bytes) -> bytes:
    """
    Generates a cryptographic signature for a given message using a secure hash algorithm and HMAC.

    Args:
        algorithm: The name of the hash algorithm (e.g., 'sha256', 'sha512').
                   This algorithm must be supported by hashlib.
        message: The message to sign (bytes).

    Returns:
        The cryptographic signature as bytes.

    Raises:
        ValueError: If the specified algorithm is unsupported by hashlib.
    """
    try:
        # Check if algorithm is supported by hashlib, hmac.new relies on this
        hash_constructor = getattr(hashlib, algorithm)
    except AttributeError:
        # Fallback for algorithms like 'md5-sha1' or others not directly attributes
        # but valid for hashlib.new()
        try:
            hashlib.new(algorithm) # Check if hashlib supports it
            # For hmac, we need the actual digestmod.
            # If it's a combined name or special case not directly on hashlib,
            # this approach might need adjustment or a mapping.
            # For standard algorithms like 'sha256', getattr is fine.
            # hmac.new can take the string name directly for digestmod in Python 3.4+
            # but getattr is more explicit for older versions or specific needs.
            # Given the context, we'll assume algorithm name is a direct hashlib attribute.
            # If hashlib.new(algorithm) works, but getattr(hashlib, algorithm) fails,
            # it implies a name that hmac.new might handle directly if it's a string.
            # However, to be safe and consistent with generate_hmac:
            if algorithm not in hashlib.algorithms_available: # More robust check
                 raise ValueError(f"Unsupported hash algorithm for signing: {algorithm}")
            # If it's available, hmac.new should be able to use its string name.
            # The previous generate_hmac used getattr(hashlib, algorithm_name)
            # which is fine for 'sha256', 'md5' etc.
            # Let's stick to that pattern for consistency.
            # The ValueError for hashlib.new() is the primary guard.
            hash_constructor = algorithm # hmac.new can take the string name
        except ValueError:
             raise ValueError(f"Unsupported hash algorithm for signing: {algorithm}")


    hmac_generator = hmac.new(secret_key, message, hash_constructor)
    return hmac_generator.digest()

if __name__ == '__main__':
    # Example usage:
    try:
        data_to_hash = "Hello, world!"
        
        sha256_hash = hash_data(data_to_hash, 'sha256')
        print(f"SHA-256 Hash: {sha256_hash}")
        
        md5_hash = hash_data(data_to_hash, 'md5')
        print(f"MD5 Hash: {md5_hash}")
        
        # Example of an unsupported algorithm
        unsupported_hash = hash_data(data_to_hash, 'sha3-256') # hashlib.new supports 'sha3_256'
        print(f"Unsupported Hash: {unsupported_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Corrected example for sha3_256 if available
        if 'sha3_256' in hashlib.algorithms_available:
            sha3_256_hash = hash_data(data_to_hash, 'sha3_256')
            print(f"SHA3-256 Hash: {sha3_256_hash}")
        else:
            print("SHA3-256 is not available in this hashlib version.")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- HMAC Examples ---")
    try:
        message_to_authenticate = b"This is a secret message."

        hmac_sha256 = generate_hmac(message_to_authenticate, 'sha256')
        print(f"HMAC-SHA256: {hmac_sha256}")

        hmac_md5 = generate_hmac(message_to_authenticate, 'md5')
        print(f"HMAC-MD5: {hmac_md5}")

        # Example of an unsupported algorithm for HMAC
        hmac_unsupported = generate_hmac(message_to_authenticate, 'sha3-256') # hashlib.new supports 'sha3_256'
        print(f"Unsupported HMAC: {hmac_unsupported}")
    except ValueError as e:
        print(f"Error: {e}")
    
    try:
        if 'sha3_256' in hashlib.algorithms_available:
            hmac_sha3_256 = generate_hmac(message_to_authenticate, 'sha3_256')
            print(f"HMAC-SHA3-256: {hmac_sha3_256}")
        else:
            print("SHA3-256 is not available for HMAC in this hashlib version.")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Signing Examples ---")
    try:
        message_to_sign = b"This is data to be signed."

        signature_sha256 = sign_data('sha256', message_to_sign)
        print(f"SHA-256 Signature (bytes): {signature_sha256}")
        print(f"SHA-256 Signature (hex): {signature_sha256.hex()}")


        signature_sha512 = sign_data('sha512', message_to_sign)
        print(f"SHA-512 Signature (bytes): {signature_sha512}")
        print(f"SHA-512 Signature (hex): {signature_sha512.hex()}")

        # Example of an unsupported algorithm for signing
        signature_unsupported = sign_data('unsupported_algo', message_to_sign)
        print(f"Unsupported Signature: {signature_unsupported}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        if 'sha3_256' in hashlib.algorithms_available:
            signature_sha3_256 = sign_data('sha3_256', message_to_sign)
            print(f"SHA3-256 Signature (bytes): {signature_sha3_256}")
            print(f"SHA3-256 Signature (hex): {signature_sha3_256.hex()}")
        else:
            print("SHA3-256 is not available for signing in this hashlib version.")
    except ValueError as e:
        print(f"Error: {e}")
