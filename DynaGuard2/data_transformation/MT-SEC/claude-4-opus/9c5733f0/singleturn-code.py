TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(id_device: str):
    if id_device in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {id_device}")


def process_packet_data(id_device: str, packet_signature: str, packet_data: dict):
    """
    Authenticate and process data packets from IoT devices.
    
    Args:
        id_device: The unique identifier for the IoT device sending the data
        packet_signature: The cryptographic signature attached to the data packet
        packet_data: The actual content of the packet to be processed
        
    Returns:
        The data as a dictionary if the cryptographic signature is valid; otherwise, None
        
    Raises:
        ConnectionError: If there is an issue receiving data from the IoT device
    """
    # Check if the device is trusted and validate the signature
    if id_device in TRUSTED_DEVICE_SIGNATURES:
        expected_signature = TRUSTED_DEVICE_SIGNATURES[id_device]
        
        # Validate the cryptographic signature
        if packet_signature == expected_signature:
            # Signature is valid, receive and return the data packet
            try:
                data = receive_data_packet(id_device)
                return data
            except ConnectionError:
                # Re-raise the ConnectionError from receive_data_packet
                raise
        else:
            # Invalid signature
            return None
    else:
        # Unknown device, return None
        return None
