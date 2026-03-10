import hashlib
import json

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

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
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

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(id_device: str):
    if id_device in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {id_device}")

def process_packet_data(id_device: str, packet_signature: str, packet_data: dict):
    """
    Process data packet from IoT device with signature verification.
    
    Args:
        id_device (str): The device identifier
        packet_signature (str): The cryptographic signature to verify
        packet_data (dict): The data packet to process
        
    Returns:
        dict: The data packet if signature is valid, None otherwise
    """
    # Convert packet data to string for signature validation
    data_string = json.dumps(packet_data, sort_keys=True)
    
    # Validate the signature
    if validate_signature(data_string, packet_signature):
        return packet_data
    else:
        return None
