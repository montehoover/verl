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


def hash_data(algo_name: str, content_data: bytes) -> bytes:
    algo_norm = algo_name.strip().lower().replace(" ", "").replace("-", "_")
    allowed = {
        "sha256": "sha256",
        "sha384": "sha384",
        "sha512": "sha512",
        "sha512_224": "sha512_224",
        "sha512_256": "sha512_256",
        "sha3_224": "sha3_224",
        "sha3_256": "sha3_256",
        "sha3_384": "sha3_384",
        "sha3_512": "sha3_512",
        "blake2b": "blake2b",
        "blake2s": "blake2s",
        "sm3": "sm3",
    }
    if algo_norm not in allowed:
        raise ValueError(
            "Unsupported or disallowed algorithm: "
            f"{algo_name}. Allowed algorithms: {', '.join(sorted(allowed))}"
        )
    try:
        hasher = hashlib.new(allowed[algo_norm])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unavailable algorithm: {algo_name}") from e
    hasher.update(bytes(content_data))
    return hasher.digest()
