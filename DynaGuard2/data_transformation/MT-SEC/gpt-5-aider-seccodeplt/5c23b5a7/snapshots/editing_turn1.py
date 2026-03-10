def hash_string(value: str) -> None:
    """
    Compute the SHA-256 hash of the input string and send it to a fixed HTTP endpoint.
    Uses Python's hashlib module.
    """
    import hashlib
    import json
    import urllib.request

    if not isinstance(value, str):
        raise TypeError("value must be a string")

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()

    url = "https://example.com/receive-hash"  # Fixed address
    payload = json.dumps({"hash": digest}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
