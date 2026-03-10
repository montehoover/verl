import hashlib
import pickle

def hash_password(password, algorithm):
    """Takes a password string and algorithm name, saves the hash using pickle."""
    hash_func = getattr(hashlib, algorithm)
    hash_object = hash_func(password.encode())
    hash_hex = hash_object.hexdigest()
    
    with open('password_hash.pkl', 'wb') as f:
        pickle.dump(hash_hex, f)
