def build_secure_url(base_url: str, path: str, security_token: str) -> str:
    """
    Build a full URL by concatenating base_url and path, and appending
    the security_token as a 'token' query parameter.

    This function performs straightforward string concatenation without
    adding or removing slashes or performing URL encoding.
    """
    return f"{base_url}{path}?token={security_token}"
