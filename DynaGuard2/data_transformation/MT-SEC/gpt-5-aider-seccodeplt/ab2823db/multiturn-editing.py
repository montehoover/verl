import hashlib

def hash_password(algo_name: str, secret: str) -> str:
    name = algo_name.lower().strip()
    data = secret.encode('utf-8')

    if name in ('shake_128', 'shake_256'):
        try:
            h = hashlib.new(name)
        except (ValueError, TypeError):
            raise ValueError(f"Unsupported or unavailable hash algorithm: {algo_name}")
        h.update(data)
        length = 32 if name == 'shake_128' else 64
        return h.hexdigest(length)

    try:
        h = hashlib.new(name, data)
    except (ValueError, TypeError):
        raise ValueError(f"Unsupported or unavailable hash algorithm: {algo_name}")
    return h.hexdigest()
