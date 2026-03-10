import hashlib

def hash_string(input_string):
    # Create SHA-256 hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    hash_hex = hash_object.hexdigest()
    
    # Send to fixed address (placeholder - you'll need to implement the actual sending mechanism)
    fixed_address = "0x1234567890abcdef"  # Replace with your actual address
    
    # This is where you would implement the actual sending logic
    # For example: send_to_address(fixed_address, hash_hex)
    
    return hash_hex
