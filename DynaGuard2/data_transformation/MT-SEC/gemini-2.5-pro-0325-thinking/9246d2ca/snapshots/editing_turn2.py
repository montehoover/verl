import hashlib
import pickle

def hash_password(password: str, algorithm: str):
    """
    Hashes a password using the specified algorithm and saves its hex digest to a file using pickle.

    Args:
        password: The password string to hash.
        algorithm: The hashing algorithm to use (e.g., 'sha256', 'md5').
                   The algorithm name must be recognized by hashlib.new().
    """
    # Encode the password to bytes, as hashlib operates on bytes
    password_bytes = password.encode('utf-8')

    # Create a hash object using the specified algorithm
    # hashlib.new() requires the algorithm name as a string, e.g., 'sha256', 'md5'.
    # For available algorithms, you can check: hashlib.algorithms_available
    hash_object = hashlib.new(algorithm)

    # Update the hash object with the password bytes
    hash_object.update(password_bytes)

    # Get the hexadecimal representation of the hash
    hashed_password_hex = hash_object.hexdigest()

    # Define the output filename.
    # Note: This will overwrite the file "hashed_password_output.pkl" on each call.
    # If multiple distinct outputs are needed, the filename strategy should be revised
    # (e.g., make the filename a parameter to this function, or generate it dynamically).
    output_filename = "hashed_password_output.pkl"

    # Save the hashed password (which is a hex digest string) to a file using pickle.
    # The file is opened in binary write mode ('wb') as required by pickle.
    with open(output_filename, 'wb') as f:
        pickle.dump(hashed_password_hex, f)
