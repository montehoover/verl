import hashlib
import pickle


def simple_hash(text: str) -> bytes:
    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    return pickle.dumps(hash_bytes, protocol=pickle.HIGHEST_PROTOCOL)
