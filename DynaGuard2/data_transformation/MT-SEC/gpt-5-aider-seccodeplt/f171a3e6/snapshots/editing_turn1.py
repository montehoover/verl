def generate_simple_hash(input_string: str) -> None:
    import hashlib
    print(hashlib.sha256(input_string.encode('utf-8')).hexdigest())
