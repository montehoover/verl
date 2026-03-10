import hashlib

def hash_string(algorithm_name: str, input_string: str) -> None:
    """
    Hashes a string using the specified algorithm and prints the hex digest.
    Assumes the algorithm_name is a supported hash algorithm in hashlib.
    """
    # Encode the string to bytes
    encoded_string = input_string.encode('utf-8')
    
    # Create a hash object using the specified algorithm
    hash_object = hashlib.new(algorithm_name)
    
    # Update the hash object with the encoded string
    hash_object.update(encoded_string)
    
    # Get the hexadecimal representation of the hash
    hex_digest = hash_object.hexdigest()
    
    # Print the hash
    print(hex_digest)
