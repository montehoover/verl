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
    # Validate the signature
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        return None
    
    if TRUSTED_DEVICE_SIGNATURES[device_id] != packet_signature:
        return None
    
    # Receive the data packet
    try:
        received_packet = receive_data_packet(device_id)
    except ConnectionError:
        raise
    
    # Process the data packet
    processed_data = {
        "device_id": device_id,
        "status": "processed",
        "data": data_packet
    }
    
    return processed_data
