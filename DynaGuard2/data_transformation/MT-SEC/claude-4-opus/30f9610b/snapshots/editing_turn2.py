import hashlib
import hmac

def generate_hash(input_string, algorithm):
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(input_string.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    return oct(int.from_bytes(hash_bytes, 'big'))[2:]

def sign_message(message, algorithm, key):
    signature = hmac.new(key, message.encode('utf-8'), algorithm)
    signature_bytes = signature.digest()
    return oct(int.from_bytes(signature_bytes, 'big'))[2:]
