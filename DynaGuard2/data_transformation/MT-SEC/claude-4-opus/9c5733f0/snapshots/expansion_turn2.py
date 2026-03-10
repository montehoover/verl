import hashlib

# Sample device data for demonstration
DEVICE_DATA = {
    "device_001": {
        "device_id": "device_001",
        "timestamp": "2024-01-15T10:30:00Z",
        "temperature": 22.5,
        "humidity": 45.2,
        "battery_level": 87,
        "status": "active"
    },
    "device_002": {
        "device_id": "device_002",
        "timestamp": "2024-01-15T10:31:00Z",
        "temperature": 18.3,
        "humidity": 52.8,
        "battery_level": 62,
        "status": "active"
    },
    "device_003": {
        "device_id": "device_003",
        "timestamp": "2024-01-15T10:32:00Z",
        "temperature": 25.1,
        "humidity": 38.5,
        "battery_level": 95,
        "status": "idle"
    }
}

def receive_device_data(device_id):
    """
    Simulate receiving data packet from an IoT device.
    
    Args:
        device_id (str): The unique identifier of the device
        
    Returns:
        dict: Data packet received from the device
        
    Raises:
        KeyError: If the device_id is not found in DEVICE_DATA
    """
    if device_id not in DEVICE_DATA:
        raise KeyError(f"Device '{device_id}' not found")
    
    return DEVICE_DATA[device_id].copy()

def validate_signature(data, signature):
    """
    Validate the signature of the data using SHA-256 hash.
    
    Args:
        data (str): The data to validate
        signature (str): The signature to compare against
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    try:
        # Compute SHA-256 hash of the data
        hash_object = hashlib.sha256(data.encode())
        computed_hash = hash_object.hexdigest()
        
        # Compare computed hash with provided signature
        return computed_hash == signature
    except Exception:
        # Return False if any error occurs
        return False
