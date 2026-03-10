import hashlib
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class HashAlgorithm:
    """
    Represents a hash algorithm, ensuring it's supported and not insecure.
    """
    INSECURE_ALGORITHMS = {'md5', 'sha1', 'md5-sha1'}

    def __init__(self, algorithm_name: str):
        """
        Initializes the HashAlgorithm.

        Args:
            algorithm_name: The name of the hash algorithm.

        Raises:
            ValueError: If the algorithm is insecure or not supported.
        """
        self.normalized_name = algorithm_name.lower()
        if self.normalized_name in self.INSECURE_ALGORITHMS:
            raise ValueError(f"Hash algorithm '{algorithm_name}' is insecure and not allowed.")
        if self.normalized_name not in hashlib.algorithms_available:
            raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported.")

    def create_hasher(self):
        """
        Creates a new hash object for this algorithm.

        Returns:
            A hash object.
        """
        try:
            return hashlib.new(self.normalized_name)
        except ValueError:
            # This should ideally be caught by the __init__ checks,
            # but kept as a fallback.
            raise ValueError(f"Hash algorithm '{self.normalized_name}' is not supported by hashlib.")

def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    """
    Computes the hash of the provided input data using a specified hash function.

    Args:
        algorithm_name: The name of the hash algorithm.
        input_data: The input data to hash.

    Returns:
        The resulting digest from the hash function.

    Raises:
        ValueError: When the chosen hash function isn't supported or is insecure.
    """
    logger.info(f"derive_hash called with algorithm: {algorithm_name}, input_data_length: {len(input_data)}")
    try:
        algo = HashAlgorithm(algorithm_name)
        hasher = algo.create_hasher()
        hasher.update(input_data)
        result = hasher.digest()
        logger.info(f"Successfully derived hash for algorithm: {algorithm_name}. Hash (hex): {result.hex()}")
        return result
    except ValueError as e:
        logger.error(f"Failed to derive hash for algorithm: {algorithm_name}. Error: {e}")
        raise
