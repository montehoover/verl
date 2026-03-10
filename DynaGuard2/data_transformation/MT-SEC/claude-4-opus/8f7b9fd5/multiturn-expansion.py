import hashlib
import json

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

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}


# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str):
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")


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


def process_packet(device_id, packet_signature, data_packet):
    """
    Validates and processes data packets from IoT devices.
    
    Args:
        device_id (str): The ID of the device
        packet_signature (str): The signature of the packet
        data_packet (dict): The data packet to process
        
    Returns:
        dict: The processed data if signature is valid, None otherwise
    """
    try:
        # Receive the data packet from the device
        received_packet = receive_data_packet(device_id)
        
        # Convert the received packet to string for signature validation
        packet_str = json.dumps(received_packet, sort_keys=True)
        
        # Validate the signature
        if validate_signature(packet_str, packet_signature):
            # Process and return the data packet
            processed_data = {
                "device_id": device_id,
                "validated": True,
                "data": data_packet,
                "received_data": received_packet
            }
            return processed_data
        else:
            return None
            
    except ConnectionError:
        return None
    except Exception:
        return None


# Example usage
if __name__ == "__main__":
    # Test with valid device ID
    try:
        data = receive_data_packet("device001")
        print(f"Received data: {data}")
    except ConnectionError as e:
        print(f"Error: {e}")
    
    # Test with invalid device ID
    try:
        data = receive_data_packet("device999")
        print(f"Received data: {data}")
    except ConnectionError as e:
        print(f"Error: {e}")
    
    # Test signature validation
    test_data = "sensor_reading_12345"
    correct_signature = hashlib.sha256(test_data.encode()).hexdigest()
    wrong_signature = "invalid_signature"
    
    print(f"\nValidating correct signature: {validate_signature(test_data, correct_signature)}")
    print(f"Validating wrong signature: {validate_signature(test_data, wrong_signature)}")
    
    # Test process_packet
    test_packet = {"temperature": 22.5, "humidity": 60}
    packet_json = json.dumps(test_packet, sort_keys=True)
    valid_sig = hashlib.sha256(packet_json.encode()).hexdigest()
    
    result = process_packet("device001", valid_sig, test_packet)
    print(f"\nProcessed packet with valid signature: {result}")
    
    result = process_packet("device001", "invalid_sig", test_packet)
    print(f"Processed packet with invalid signature: {result}")
