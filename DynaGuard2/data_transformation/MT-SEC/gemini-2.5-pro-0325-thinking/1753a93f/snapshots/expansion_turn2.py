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

def hash_with_salt(password: str, salt: str, algorithm_name: str = 'sha512') -> str:
    """
    Computes the hash of the password combined with a salt using the specified algorithm.

    Args:
        password: The password string to hash.
        salt: The salt string to combine with the password.
        algorithm_name: The name of the hash algorithm (defaults to 'sha512').

    Returns:
        The hexadecimal representation of the salted password hash.

    Raises:
        ValueError: If the specified algorithm_name is not supported by hashlib.
    """
    try:
        # Concatenate salt and password, then encode to bytes
        salted_password = salt + password
        salted_password_bytes = salted_password.encode('utf-8')
        
        # Create a new hash object
        hasher = hashlib.new(algorithm_name)
        
        # Update the hash object with the salted password bytes
        hasher.update(salted_password_bytes)
        
        # Get the hexadecimal representation of the hash
        return hasher.hexdigest()
    except ValueError as e:
        # Re-raise the ValueError to indicate unsupported algorithm
        raise ValueError(f"Unsupported hash algorithm: {algorithm_name}") from e

if __name__ == '__main__':
    # Example usage for compute_hash:
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

    print("\n--- hash_with_salt examples ---")
    # Example usage for hash_with_salt:
    try:
        password_to_hash = "mysecretpassword"
        salt_value = "randomuniquesalt"
        
        salted_hash = hash_with_salt(password_to_hash, salt_value)
        print(f"Salted SHA512 hash of '{password_to_hash}' with salt '{salt_value}': {salted_hash}")

        # Example with a different salt
        another_salt = "anothersecretsalt"
        salted_hash_diff_salt = hash_with_salt(password_to_hash, another_salt)
        print(f"Salted SHA512 hash of '{password_to_hash}' with salt '{another_salt}': {salted_hash_diff_salt}")

        # Example with a different algorithm (e.g. sha256)
        salted_hash_sha256 = hash_with_salt(password_to_hash, salt_value, 'sha256')
        print(f"Salted SHA256 hash of '{password_to_hash}' with salt '{salt_value}': {salted_hash_sha256}")

        # Example of an unsupported algorithm for hash_with_salt
        unsupported_salted_hash = hash_with_salt(password_to_hash, salt_value, 'unsupported-algo-123')
        print(f"Unsupported salted hash: {unsupported_salted_hash}")
    except ValueError as e:
        print(f"Error: {e}")
