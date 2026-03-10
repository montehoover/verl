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

def handle_data_packet(device_id: str, signature: str, data: dict):
    try:
        # Attempt to receive the data packet
        received_data = receive_data_packet(device_id)
        
        # Verify the signature
        if device_id in TRUSTED_DEVICE_SIGNATURES and TRUSTED_DEVICE_SIGNATURES[device_id] == signature:
            # Signature is valid, return the data
            return data
        else:
            # Invalid signature
            return None
            
    except ConnectionError:
        # Re-raise the ConnectionError if packet reception fails
        raise
