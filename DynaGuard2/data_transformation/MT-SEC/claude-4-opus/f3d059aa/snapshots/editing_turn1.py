import re

def append_user_header(user_header: str) -> str:
    return f"Content-Type: {user_header}"
