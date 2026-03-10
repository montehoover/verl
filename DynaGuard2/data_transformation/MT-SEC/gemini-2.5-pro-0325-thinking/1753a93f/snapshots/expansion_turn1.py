import hashlib

def compute_hash(data: str, algorithm_name: str) -> str:
    """
    Computes the hash of the input data using the specified algorithm.

    Args:
        data: The string input to hash.
        algorithm_name: The name of the hash algorithm (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal representation of the hash.

    Raises:
        ValueError: If the specified algorithm_name is not supported by hashlib.
    """
    try:
        # Encode the input string to bytes, as hashlib works on bytes
        data_bytes = data.encode('utf-8')
        
        # Create a new hash object
        # hashlib.new() will raise ValueError if the algorithm is not supported
        hasher = hashlib.new(algorithm_name)
        
        # Update the hash object with the data
        hasher.update(data_bytes)
        
        # Get the hexadecimal representation of the hash
        return hasher.hexdigest()
    except ValueError as e:
        # Re-raise the ValueError to indicate unsupported algorithm
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e

if __name__ == '__main__':
    # Example usage:
    try:
        data_to_hash = "hello world"
        
        sha256_hash = compute_hash(data_to_hash, 'sha256')
        print(f"SHA256 hash of '{data_to_hash}': {sha256_hash}")
        
        md5_hash = compute_hash(data_to_hash, 'md5')
        print(f"MD5 hash of '{data_to_hash}': {md5_hash}")

        # Example of an unsupported algorithm
        unsupported_hash = compute_hash(data_to_hash, 'sha3-unknown')
        print(f"Unsupported hash: {unsupported_hash}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example with a different algorithm
        sha512_hash = compute_hash("another piece of data", "sha512")
        print(f"SHA512 hash of 'another piece of data': {sha512_hash}")
    except ValueError as e:
        print(f"Error: {e}")
    
    try:
        # Example of an invalid algorithm name that hashlib might not directly list
        # but is still invalid.
        invalid_algo_hash = compute_hash(data_to_hash, 'myownhash')
        print(f"Invalid algo hash: {invalid_algo_hash}")
    except ValueError as e:
        print(f"Error: {e}")
