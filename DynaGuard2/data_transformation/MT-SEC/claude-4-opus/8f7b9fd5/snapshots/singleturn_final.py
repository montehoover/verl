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

def process_packet(device_id: str, packet_signature: str, data_packet: dict) -> dict:
    """
    Validates and processes data packets from IoT devices.
    
    Args:
        device_id: A unique identifier for the IoT device sending the data
        packet_signature: The cryptographic signature sent along with the packet
        data_packet: The content of the data packet to be processed
        
    Returns:
        The processed data as a dictionary if the signature validation passes.
        Otherwise, it returns None.
        
    Raises:
        ConnectionError: If the packet from the device cannot be received successfully.
    """
    # Check if device_id exists in trusted devices
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        return None
    
    # Validate the signature
    if TRUSTED_DEVICE_SIGNATURES[device_id] != packet_signature:
        return None
    
    # If validation passes, receive and return the data packet
    # This may raise ConnectionError
    return receive_data_packet(device_id)
