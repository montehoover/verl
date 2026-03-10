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
    # Check if the device is in the trusted devices list
    if id_device not in TRUSTED_DEVICE_SIGNATURES:
        raise ConnectionError(f"Failed to receive packet from device {id_device}")
    
    # Validate the cryptographic signature
    if TRUSTED_DEVICE_SIGNATURES[id_device] == packet_signature:
        # Signature is valid, return the packet data
        return packet_data
    else:
        # Invalid signature, return None
        return None
