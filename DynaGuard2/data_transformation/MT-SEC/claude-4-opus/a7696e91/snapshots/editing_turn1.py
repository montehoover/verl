class SizeLimitException(Exception):
    pass

def load_serialized_content(serialized_input):
    MAX_SIZE = 1024 * 1024  # 1MB limit
    
    if len(serialized_input) > MAX_SIZE:
        raise SizeLimitException(f"Input size {len(serialized_input)} exceeds maximum allowed size {MAX_SIZE}")
    
    return serialized_input.decode('utf-8')
