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

def verify_signature(device_id: str, signature: str, trusted_signatures: dict = TRUSTED_DEVICE_SIGNATURES) -> bool:
    """
    Pure function to verify that the provided signature matches the trusted signature
    for the given device_id.

    Args:
        device_id: Identification number of the IoT device.
        signature: Cryptographic signature accompanying the packet.
        trusted_signatures: Mapping of device IDs to their trusted signatures.

    Returns:
        True if the signature matches the trusted signature for the device; otherwise, False.
    """
    expected = trusted_signatures.get(device_id)
    return expected is not None and signature == expected

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

    # Verify the signature using the pure function.
    if verify_signature(device_id, signature):
        return data

    return None
