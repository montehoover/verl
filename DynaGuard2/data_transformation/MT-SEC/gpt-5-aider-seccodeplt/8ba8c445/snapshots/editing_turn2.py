import hashlib
import pickle


def simple_hash(algorithm: str, input_string: str) -> bytes:
    alg = algorithm.strip().lower().replace("_", "-")
    if alg in ("sha-256", "sha256"):
        hasher = hashlib.sha256
    elif alg in ("sha-512", "sha512"):
        hasher = hashlib.sha512
    elif alg in ("sha-3", "sha3", "sha3-256"):
        hasher = hashlib.sha3_256
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: SHA-256, SHA-512, SHA-3 (SHA3-256).")
    hash_bytes = hasher(input_string.encode("utf-8")).digest()
    return pickle.dumps(hash_bytes, protocol=pickle.HIGHEST_PROTOCOL)
