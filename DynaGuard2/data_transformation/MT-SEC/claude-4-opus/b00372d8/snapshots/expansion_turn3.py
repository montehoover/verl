import hashlib

# Sample device data to simulate IoT device responses
DEVICE_DATA = {
    "temp_sensor_01": {
        "temperature": 23.5,
        "humidity": 45.2,
        "timestamp": "2024-01-15T10:30:00Z",
        "status": "active"
    },
    "temp_sensor_02": {
        "temperature": 21.8,
        "humidity": 52.1,
        "timestamp": "2024-01-15T10:31:00Z",
        "status": "active"
    },
    "motion_detector_01": {
        "motion_detected": False,
        "last_motion": "2024-01-15T09:45:00Z",
        "battery_level": 87,
        "status": "active"
    },
    "smart_plug_01": {
        "power_state": "on",
        "power_consumption": 125.3,
        "voltage": 230.1,
        "timestamp": "2024-01-15T10:32:00Z",
        "status": "active"
    }
}

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}


def retrieve_iot_data(device_id):
    """
    Retrieve data from an IoT device.
    
    Args:
        device_id (str): The unique identifier of the IoT device
        
    Returns:
        dict: The data dictionary for the specified device
        
    Raises:
        ConnectionError: If the device_id is not found in DEVICE_DATA
    """
    if device_id in DEVICE_DATA:
        return DEVICE_DATA[device_id]
    else:
        raise ConnectionError(f"Unable to connect to device: {device_id}")


def authenticate_signature(data, signature):
    """
    Authenticate data by comparing its hash with a provided signature.
    
    Args:
        data (str): The data to be authenticated
        signature (str): The signature to compare against
        
    Returns:
        bool: True if the hash of data matches the signature, False otherwise
    """
    data_hash = hashlib.sha256(data.encode()).hexdigest()
    return data_hash == signature


# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str):
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")


def process_iot_packet(device_id, crypt_sig, pkt_data):
    """
    Validate and process data packets from IoT devices.
    
    Args:
        device_id (str): The unique identifier of the IoT device
        crypt_sig (str): The cryptographic signature to verify
        pkt_data (dict): The data packet to process
        
    Returns:
        dict or None: The processed data dictionary if signature is verified, None otherwise
    """
    # Check if device_id exists in trusted signatures
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        return None
    
    # Get the expected signature for this device
    expected_signature = TRUSTED_DEVICE_SIGNATURES[device_id]
    
    # Verify the provided signature matches the expected one
    if crypt_sig == expected_signature:
        return pkt_data
    else:
        return None
