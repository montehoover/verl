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


def handle_data_packet(device_id: str, signature: str, data: dict) -> dict:
    """
    Process data packets received from IoT devices after verifying cryptographic signature.
    
    Args:
        device_id: The identification number of the IoT device that sent the data
        signature: The cryptographic signature that accompanies the packet
        data: The payload of the packet that needs to be processed
        
    Returns:
        The data as a dictionary if the signature is valid, otherwise None
        
    Raises:
        ConnectionError: If the data packet is not received properly
    """
    # Check if the device_id exists in trusted signatures
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        return None
    
    # Verify the signature matches the trusted signature for this device
    if TRUSTED_DEVICE_SIGNATURES[device_id] != signature:
        return None
    
    # If signature is valid, receive the data packet
    # This may raise ConnectionError if reception fails
    received_data = receive_data_packet(device_id)
    
    return received_data
