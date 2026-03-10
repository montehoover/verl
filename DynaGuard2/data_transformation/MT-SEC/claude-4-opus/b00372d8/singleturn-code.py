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


def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: dict):
    """
    Validates and processes data packets from IoT devices.
    
    Args:
        device_id: str, a unique identifier for the IoT device sending the data
        crypt_sig: str, the cryptographic signature sent along with the packet
        pkt_data: dict, the content of the data packet to be processed
    
    Returns:
        The processed data as a dictionary if the signature validation passes.
        Otherwise, it returns None.
    
    Raises:
        ConnectionError: if the packet from the device cannot be received successfully
    """
    # Validate the cryptographic signature
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        expected_signature = TRUSTED_DEVICE_SIGNATURES[device_id]
        if crypt_sig == expected_signature:
            # Signature is valid, receive and return the data packet
            processed_data = receive_data_packet(device_id)
            return processed_data
        else:
            # Invalid signature
            return None
    else:
        # Unknown device ID
        return None
