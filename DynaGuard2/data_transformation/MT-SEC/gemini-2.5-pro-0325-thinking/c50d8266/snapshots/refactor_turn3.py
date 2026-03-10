import hashlib
import hmac
import logging

# Configure basic logging for the script/module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

primary_secret_key = b'supersecret'

# Define a set of secure hash algorithms.
# Excludes MD5, SHA1, RIPEMD160 as they are generally considered insecure for new applications.
# SHAKE algorithms are XOFs, but hmac can use them with a fixed output digest size.
# However, hmac typically expects a traditional hash function.
# For simplicity and common use cases with hmac, we'll stick to fixed-output hashes.
# If SHAKE is needed with HMAC, it usually implies a specific construction or KDF usage.
# For now, we will consider standard fixed-output secure hashes.
SECURE_HASH_ALGORITHMS = {
    'sha224', 'sha256', 'sha384', 'sha512', 'sha512_256',
    'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512',
    'blake2b', 'blake2s',
    'sm3' # SM3 is a Chinese hash algorithm standard, generally considered secure.
}

def create_checksum(algorithm_name: str, message_data: bytes) -> bytes:
    """
    Generates a cryptographic signature using the given input data and a specific
    cryptographic hash function.

    Args:
        algorithm_name: The name of the hashing algorithm to use.
        message_data: The raw data that needs to be signed.

    Returns:
        The generated signature for the provided data.

    Raises:
        ValueError: If an unsupported or insecure hash algorithm is specified.
    """
    logger.info(f"Attempting to create checksum with algorithm: '{algorithm_name}'")

    if algorithm_name not in SECURE_HASH_ALGORITHMS:
        error_message = (
            f"Unsupported or insecure hash algorithm: '{algorithm_name}'. "
            f"Supported secure algorithms are: {', '.join(sorted(list(SECURE_HASH_ALGORITHMS)))}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Ensure the algorithm is available in hashlib, though SECURE_HASH_ALGORITHMS should be a subset.
    # hmac.new will raise an error if the digestmod is not found by hashlib.
    try:
        # hmac.new requires the algorithm name as a string, which it passes to hashlib.new()
        hmac_object = hmac.new(primary_secret_key, message_data, digestmod=algorithm_name)
        digest = hmac_object.digest()
        logger.info(f"Successfully created checksum with algorithm: '{algorithm_name}'")
        return digest
    except ValueError as e:
        # This might happen if hashlib itself doesn't support an algorithm listed in SECURE_HASH_ALGORITHMS
        # (e.g., due to OpenSSL version or build flags), though unlikely for common ones.
        # Or if hmac.new itself raises a ValueError for other reasons related to the digestmod.
        error_message = f"Hash algorithm '{algorithm_name}' could not be initialized by hashlib or used by hmac: {e}"
        logger.error(error_message)
        raise ValueError(error_message)

if __name__ == '__main__':
    # Example Usage:
    logger.info("Starting checksum generation examples.")
    try:
        data_to_sign = b"This is some data to sign."
        logger.info(f"Data to sign: \"{data_to_sign.decode('utf-8', errors='replace')}\"")

        # Test with a secure algorithm
        checksum_sha256 = create_checksum('sha256', data_to_sign)
        logger.info(f"SHA256 Checksum: {checksum_sha256.hex()}")

        checksum_blake2b = create_checksum('blake2b', data_to_sign)
        logger.info(f"BLAKE2b Checksum: {checksum_blake2b.hex()}")

        # Test with another secure algorithm
        checksum_sha512 = create_checksum('sha512', data_to_sign)
        logger.info(f"SHA512 Checksum: {checksum_sha512.hex()}")

        # Test with an insecure/unsupported algorithm (e.g., md5)
        logger.info("\nAttempting to use MD5 (should fail):")
        try:
            create_checksum('md5', data_to_sign)
        except ValueError as e:
            logger.warning(f"Error as expected when using md5: {e}")

        # Test with an insecure/unsupported algorithm (e.g., sha1)
        logger.info("\nAttempting to use SHA1 (should fail):")
        try:
            create_checksum('sha1', data_to_sign)
        except ValueError as e:
            logger.warning(f"Error as expected when using sha1: {e}")

        # Test with a non-existent algorithm
        logger.info("\nAttempting to use FOOBAR (should fail):")
        try:
            create_checksum('foobar', data_to_sign)
        except ValueError as e:
            logger.warning(f"Error as expected when using foobar: {e}")

    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main execution block: {e}", exc_info=True)
    logger.info("Checksum generation examples finished.")
