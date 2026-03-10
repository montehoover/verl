import hashlib

# Example device data dictionary for demonstration
DEVICE_DATA = {
    "device_001": {
        "temperature": 23.5,
        "humidity": 65,
        "timestamp": "2024-01-15T10:30:00",
        "battery": 87
    },
    "device_002": {
        "temperature": 21.2,
        "humidity": 70,
        "timestamp": "2024-01-15T10:32:00",
        "battery": 92
    },
    "device_003": {
        "temperature": 25.8,
        "humidity": 58,
        "timestamp": "2024-01-15T10:28:00",
        "battery": 45
    }
}


def receive_data_packet(device_id):
    """
    Retrieves a data packet from the DEVICE_DATA dictionary for a given device ID.
    
    Args:
        device_id (str): The ID of the device to retrieve data from
        
    Returns:
        dict: The data packet for the specified device
        
    Raises:
        ConnectionError: If the device ID is not found in DEVICE_DATA
    """
    if device_id not in DEVICE_DATA:
        raise ConnectionError(f"Device {device_id} not found or not connected")
    
    return DEVICE_DATA[device_id]


def validate_signature(data, signature):
    """
    Validates the signature of the given data using SHA-256 hash.
    
    Args:
        data (str): The data to validate
        signature (str): The signature to compare against
        
    Returns:
        bool: True if signature matches, False otherwise
    """
    try:
        # Check if inputs are strings
        if not isinstance(data, str) or not isinstance(signature, str):
            return False
        
        # Compute SHA-256 hash of the data
        data_hash = hashlib.sha256(data.encode()).hexdigest()
        
        # Compare computed hash with provided signature
        return data_hash == signature
        
    except Exception:
        # Return False for any unsupported operations or errors
        return False


# Example usage
if __name__ == "__main__":
    # Test with valid device ID
    try:
        data = receive_data_packet("device_001")
        print(f"Received data: {data}")
    except ConnectionError as e:
        print(f"Error: {e}")
    
    # Test with invalid device ID
    try:
        data = receive_data_packet("device_999")
        print(f"Received data: {data}")
    except ConnectionError as e:
        print(f"Error: {e}")
    
    # Test signature validation
    test_data = "sensor_reading_12345"
    correct_signature = hashlib.sha256(test_data.encode()).hexdigest()
    wrong_signature = "invalid_signature"
    
    print(f"\nValidating correct signature: {validate_signature(test_data, correct_signature)}")
    print(f"Validating wrong signature: {validate_signature(test_data, wrong_signature)}")
