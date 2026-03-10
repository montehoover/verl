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


def process_packet(device_id: str, packet_signature: str, data_packet: dict):
    """
    Validates and processes a data packet from an IoT device.

    - Verifies the cryptographic signature against trusted signatures.
    - Ensures the packet can be received (simulated via receive_data_packet).
    - Returns processed data dict if signature is valid; otherwise returns None.
    - Raises ConnectionError if the device's packet cannot be received successfully.
    """
    # Validate signature before handling data
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(device_id)
    if expected_signature is None or packet_signature != expected_signature:
        return None

    # Ensure the packet from the device can be received (may raise ConnectionError)
    receive_data_packet(device_id)

    # Process and return the data (here we simply add metadata and return a copy)
    processed = dict(data_packet)
    processed["device_id"] = device_id
    return processed
