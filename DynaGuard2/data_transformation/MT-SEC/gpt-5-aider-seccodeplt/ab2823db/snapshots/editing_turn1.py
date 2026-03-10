import hashlib

def simple_hash(s: str) -> None:
    print(hashlib.sha256(s.encode('utf-8')).hexdigest())
