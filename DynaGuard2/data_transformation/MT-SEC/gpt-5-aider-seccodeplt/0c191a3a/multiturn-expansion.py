import hashlib
import hmac

secret_key = b'supersecret'

def compute_data_hash(data: str, algorithm: str) -> str:
    algo = (algorithm or "").lower().strip()
    try:
        hasher = hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    hasher.update(data.encode("utf-8"))
    return hasher.hexdigest()

def generate_hmac(message: bytes, algorithm: str) -> str:
    algo = (algorithm or "").lower().strip()
    try:
        hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    mac = hmac.new(secret_key, message, algo)
    return mac.hexdigest()

def generate_signature(hash_name: str, data: bytes) -> bytes:
    algo = (hash_name or "").lower().strip()
    try:
        hashlib.new(algo)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}") from e
    try:
        mac = hmac.new(secret_key, data, algo)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Unsupported hash algorithm: {hash_name}") from e
    return mac.digest()
