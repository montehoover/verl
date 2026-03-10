import hashlib
import logging
from dataclasses import dataclass
from typing import Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HashAlgorithmConfig:
    """Configuration for hash algorithm validation and management."""
    insecure_algorithms: Set[str] = None
    
    def __post_init__(self):
        if self.insecure_algorithms is None:
            self.insecure_algorithms = {'md5', 'sha1', 'md5-sha1'}
    
    def is_algorithm_allowed(self, algorithm_name: str) -> bool:
        """Check if the algorithm is allowed (not insecure)."""
        return algorithm_name.lower() not in self.insecure_algorithms
    
    def is_algorithm_supported(self, algorithm_name: str) -> bool:
        """Check if the algorithm is supported by hashlib."""
        return algorithm_name in hashlib.algorithms_available

# Global configuration instance
hash_config = HashAlgorithmConfig()

def derive_hash(algorithm_name: str, input_data: bytes) -> bytes:
    logger.info(f"derive_hash called with algorithm: {algorithm_name}, input_data length: {len(input_data)} bytes")
    
    # Check if the algorithm is allowed (not insecure)
    if not hash_config.is_algorithm_allowed(algorithm_name):
        logger.error(f"Attempted to use insecure hash algorithm: {algorithm_name}")
        raise ValueError(f"Insecure hash algorithm '{algorithm_name}' is not supported")
    
    # Check if the algorithm is supported by hashlib
    if not hash_config.is_algorithm_supported(algorithm_name):
        logger.error(f"Attempted to use unsupported hash algorithm: {algorithm_name}")
        raise ValueError(f"Hash algorithm '{algorithm_name}' is not supported")
    
    # Create hash object and compute digest
    hash_obj = hashlib.new(algorithm_name, input_data)
    result = hash_obj.digest()
    
    logger.info(f"Successfully computed {algorithm_name} hash, output length: {len(result)} bytes")
    return result
