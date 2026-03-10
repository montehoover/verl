import hashlib

def hash_string(algorithm_name, input_string):
    # Get the hash algorithm from hashlib
    hash_algorithm = getattr(hashlib, algorithm_name)
    
    # Create hash of the input string
    hash_object = hash_algorithm(input_string.encode())
    hash_hex = hash_object.hexdigest()
    
    # Print the hash
    print(hash_hex)
    
    # Send to fixed address (placeholder - you'll need to implement the actual sending mechanism)
    fixed_address = "0x1234567890abcdef"  # Replace with your actual address
    
    # This is where you would implement the actual sending logic
    # For example: send_to_address(fixed_address, hash_hex)
    
    return hash_hex
