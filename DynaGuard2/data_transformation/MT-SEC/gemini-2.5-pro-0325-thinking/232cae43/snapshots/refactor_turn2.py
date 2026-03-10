import hashlib
import hmac

def compute_hmac(hash_algorithm: str, payload: bytes, secret: bytes) -> bytes:
    """
    Computes the HMAC signature of the data.

    Args:
        hash_algorithm: The hash function to apply.
        payload: The input to be signed.
        secret: The secret key used for signing with the hash function.

    Returns:
        The computed HMAC signature.

    Raises:
        ValueError: If the hash function is not supported by hashlib.
    """
    # Validate that the requested hash algorithm is supported by hashlib.
    # This check ensures that `getattr` will find a corresponding hash constructor
    # and prevents potential errors if an unsupported algorithm name is passed.
    if hash_algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

    # Dynamically retrieve the hash constructor (e.g., hashlib.sha256) 
    # from the hashlib module based on the hash_algorithm string.
    hash_constructor = getattr(hashlib, hash_algorithm)

    # Create the HMAC object using the provided secret key, message payload,
    # and the retrieved hash constructor.
    h = hmac.new(
        key=secret,
        msg=payload,
        digestmod=hash_constructor
    )

    # Compute and return the binary digest of the HMAC.
    return h.digest()
