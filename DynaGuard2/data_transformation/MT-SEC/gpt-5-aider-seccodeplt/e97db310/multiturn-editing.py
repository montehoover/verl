VALID_SIGNATURES = {
    "/tmp/example.txt": "5d41402abc4b2a76b9719d911017c592",
    "/tmp/sample.txt": "2dcd14d9f59bb42477979fdb6e67e86a"
}

for file_name in VALID_SIGNATURES.keys():
    with open(file_name, 'w') as file:
        file.write("")

def load_file(file_name: str, signature: str) -> int:
    """
    Verify a file's cryptographic signature against a known-good list.

    Args:
        file_name: The name or path of the file to verify.
        signature: The cryptographic signature of the file.

    Returns:
        1 if the signature is valid for the given file; otherwise 0.
    """
    expected = VALID_SIGNATURES.get(file_name)
    if expected is None:
        return 0
    return 1 if signature == expected else 0
