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
    """
    Process a data packet from an IoT device.

    - Attempts to receive the packet using receive_data_packet.
    - Verifies the cryptographic signature for the given device_id.
    - Returns the received data dict if the signature is valid.
    - Returns None if the signature is invalid.
    - Raises ConnectionError if the packet is not received properly.
    """
    # Attempt to receive the data packet; propagate ConnectionError on failure
    received = receive_data_packet(device_id)

    # Verify the signature against trusted signatures
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(device_id)
    if expected_signature is None:
        # Shouldn't happen with current receive_data_packet, but guard anyway
        return None

    if signature != expected_signature:
        return None

    # Signature is valid; return the received packet data
    return received
