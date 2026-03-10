import re

def append_user_header(user_header: str) -> str:
    # Validate the header value to ensure it doesn't contain invalid characters
    # HTTP header values should not contain control characters or newlines
    if re.search(r'[\r\n\x00-\x1f\x7f]', user_header):
        user_header = re.sub(r'[\r\n\x00-\x1f\x7f]', '', user_header)
    
    # Create the full header string with the custom header
    # Using a generic header name "X-Custom-Header" for the custom header
    full_header = f"X-Custom-Header: {user_header}"
    
    return full_header
