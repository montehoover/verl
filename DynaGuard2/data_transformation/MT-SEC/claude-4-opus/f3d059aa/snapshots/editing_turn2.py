import re

def append_user_header(user_header: str) -> str:
    # Remove any characters that are not alphanumeric, dashes, or underscores
    cleansed_header = re.sub(r'[^a-zA-Z0-9\-_]', '', user_header)
    return f"X-Custom-Header: {cleansed_header}"
