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

def validate_signature(device_id: str, crypt_sig: str) -> bool:
    """Validate the cryptographic signature for a given device."""
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        return False
    return TRUSTED_DEVICE_SIGNATURES[device_id] == crypt_sig

def handle_packet_data(device_id: str, pkt_data: dict, received_data: dict) -> dict:
    """Process and combine packet data with received data."""
    return {
        "device_id": device_id,
        "data": pkt_data,
        "received_data": received_data,
        "status": "validated"
    }

def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: dict):
    # Validate the cryptographic signature
    if not validate_signature(device_id, crypt_sig):
        return None
    
    # Receive the data packet (this may raise ConnectionError)
    received_data = receive_data_packet(device_id)
    
    # Process the packet data
    processed_data = handle_packet_data(device_id, pkt_data, received_data)
    
    return processed_data
