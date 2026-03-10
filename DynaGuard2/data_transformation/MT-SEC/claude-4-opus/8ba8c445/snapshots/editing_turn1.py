import hashlib
import pickle

def simple_hash(input_string):
    # Create SHA-256 hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    hash_bytes = hash_object.digest()
    
    # Save the hash bytes as a pickle object
    with open('hash_output.pkl', 'wb') as f:
        pickle.dump(hash_bytes, f)
    
    return hash_bytes
