import hashlib
import pickle

def simple_hash(algorithm, input_string):
    # Create hash based on the specified algorithm
    if algorithm == 'SHA-256':
        hash_object = hashlib.sha256(input_string.encode())
    elif algorithm == 'SHA-512':
        hash_object = hashlib.sha512(input_string.encode())
    elif algorithm == 'SHA-3':
        hash_object = hashlib.sha3_256(input_string.encode())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    hash_bytes = hash_object.digest()
    
    # Save the hash bytes as a pickle object
    with open('hash_output.pkl', 'wb') as f:
        pickle.dump(hash_bytes, f)
    
    return hash_bytes
