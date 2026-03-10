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
    Process a data packet received from an IoT device.

    Args:
        device_id: Identification number of the IoT device.
        signature: Cryptographic signature accompanying the packet.
        data: Payload of the packet to be processed.

    Returns:
        The data dictionary if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If the data packet is not received properly.
    """
    # Attempt to receive a packet to confirm proper reception/connectivity.
    # This will raise ConnectionError if the device is not reachable/valid.
    receive_data_packet(device_id)

    # Verify the signature against the trusted signature for this device.
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(device_id)
    if expected_signature is None:
        # No trusted signature for this device; treat as invalid signature.
        return None

    if signature == expected_signature:
        return data

    return None
