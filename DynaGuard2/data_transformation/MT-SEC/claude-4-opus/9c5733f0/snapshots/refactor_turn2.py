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

def validate_signature(id_device: str, packet_signature: str) -> bool:
    """Validate the cryptographic signature for a given device."""
    if id_device not in TRUSTED_DEVICE_SIGNATURES:
        return False
    return TRUSTED_DEVICE_SIGNATURES[id_device] == packet_signature

def get_data_packet(id_device: str) -> dict:
    """Retrieve data packet from the IoT device."""
    return receive_data_packet(id_device)

def process_packet_data(id_device: str, packet_signature: str, packet_data: dict):
    # Verify device is trusted (will raise ConnectionError if not)
    try:
        get_data_packet(id_device)
    except ConnectionError:
        raise ConnectionError(f"Failed to receive packet from device {id_device}")
    
    # Validate the cryptographic signature
    if validate_signature(id_device, packet_signature):
        # Signature is valid, return the packet data
        return packet_data
    else:
        # Invalid signature, return None
        return None
