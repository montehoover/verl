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
    # Validate the cryptographic signature
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        return None
    
    if TRUSTED_DEVICE_SIGNATURES[device_id] != crypt_sig:
        return None
    
    # Receive the data packet (this may raise ConnectionError)
    received_data = receive_data_packet(device_id)
    
    # Process the packet data
    processed_data = {
        "device_id": device_id,
        "data": pkt_data,
        "received_data": received_data,
        "status": "validated"
    }
    
    return processed_data
