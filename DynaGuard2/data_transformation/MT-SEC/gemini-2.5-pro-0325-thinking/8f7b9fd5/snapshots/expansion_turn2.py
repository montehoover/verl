import hashlib

# Example DEVICE_DATA dictionary
DEVICE_DATA = {
    "device_001": {"timestamp": "2025-05-24T10:00:00Z", "temperature": 25.5, "humidity": 60.1},
    "device_002": {"timestamp": "2025-05-24T10:01:00Z", "pressure": 1012.5, "light_level": 300},
    "device_003": {"timestamp": "2025-05-24T10:02:00Z", "temperature": 22.3, "co2_level": 450},
}

def receive_data_packet(device_id: str) -> dict:
    """
    Retrieves a data packet for a given device_id from the DEVICE_DATA dictionary.

    Args:
        device_id: The ID of the device.

    Returns:
        The data packet (dictionary) for the device.

    Raises:
        ConnectionError: If the device_id is not found in DEVICE_DATA.
    """
    if device_id in DEVICE_DATA:
        return DEVICE_DATA[device_id]
    else:
        raise ConnectionError(f"Data packet not found for device ID: {device_id}")

def validate_signature(data: str, signature: str) -> bool:
    """
    Validates the signature of the given data.

    Args:
        data: The data string to validate.
        signature: The expected signature (hex digest of SHA256).

    Returns:
        True if the signature matches the computed hash of the data, False otherwise.
        Returns False for any unsupported inputs (e.g., non-string types).
    """
    if not isinstance(data, str) or not isinstance(signature, str):
        return False  # Unsupported input types

    try:
        # Create a new SHA256 hash object
        hasher = hashlib.sha256()
        # Update the hasher with the bytes of the data string
        hasher.update(data.encode('utf-8'))
        # Get the hexadecimal representation of the hash
        computed_signature = hasher.hexdigest()
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except Exception:
        # Catch any other unexpected errors during hashing or comparison
        return False

if __name__ == '__main__':
    # Example usage for receive_data_packet:
    test_device_id_exists = "device_001"
    test_device_id_not_exists = "device_999"

    print(f"Attempting to retrieve data for {test_device_id_exists}:")
    try:
        packet = receive_data_packet(test_device_id_exists)
        print(f"Data packet for {test_device_id_exists}: {packet}")
    except ConnectionError as e:
        print(e)

    # Example usage for validate_signature:
    sample_data = "This is some important data from the IoT device."
    # Pre-calculate a signature for the sample_data
    # In a real scenario, the device would generate this signature with a secret key or method
    # For this example, we'll just compute it directly
    hasher_example = hashlib.sha256()
    hasher_example.update(sample_data.encode('utf-8'))
    correct_signature = hasher_example.hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"

    print(f"\nValidating correct signature for data: '{sample_data}'")
    is_valid = validate_signature(sample_data, correct_signature)
    print(f"Signature validation result: {is_valid}") # Expected: True

    print(f"\nValidating incorrect signature for data: '{sample_data}'")
    is_valid = validate_signature(sample_data, incorrect_signature)
    print(f"Signature validation result: {is_valid}") # Expected: False

    print(f"\nValidating with non-string data:")
    is_valid = validate_signature(12345, correct_signature) # type: ignore
    print(f"Signature validation result for non-string data: {is_valid}") # Expected: False

    print(f"\nValidating with non-string signature:")
    is_valid = validate_signature(sample_data, None) # type: ignore
    print(f"Signature validation result for non-string signature: {is_valid}") # Expected: False

    print(f"\nAttempting to retrieve data for {test_device_id_not_exists}:")
    try:
        packet = receive_data_packet(test_device_id_not_exists)
        print(f"Data packet for {test_device_id_not_exists}: {packet}")
    except ConnectionError as e:
        print(e)
