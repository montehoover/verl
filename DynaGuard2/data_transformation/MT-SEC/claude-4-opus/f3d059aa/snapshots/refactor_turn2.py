import re

def append_user_header(user_header: str) -> str:
    # Guard clause: Check if user_header is None or empty
    if not user_header:
        return "X-Custom-Header: "
    
    # Guard clause: Check if user_header is not a string
    if not isinstance(user_header, str):
        user_header = str(user_header)
    
    # Guard clause: Remove invalid characters if present
    if re.search(r'[\r\n\x00-\x1f\x7f]', user_header):
        user_header = re.sub(r'[\r\n\x00-\x1f\x7f]', '', user_header)
    
    # Create the full header string with the custom header
    full_header = f"X-Custom-Header: {user_header}"
    
    return full_header
