TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str) -> dict:
    """
    Simulates receiving a data packet from an IoT device.
    """
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        # Example data for a trusted device
        return {"temperature": 22.5, "humidity": 60}
    else:
        # This device is not recognized or there's a connection issue.
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

def process_packet(device_id: str, packet_signature: str, data_packet: dict) -> dict | None:
    """
    Validates and processes data packets from IoT devices.

    Each packet is accompanied by a cryptographic signature that must be
    authenticated before handling the data. The actual data is fetched
    by calling receive_data_packet after successful authentication.

    Args:
        device_id: A unique identifier for the IoT device sending the data.
        packet_signature: The cryptographic signature sent along with the packet.
        data_packet: The content of the data packet to be processed.
                     (Note: Based on the provided example, this argument is not
                      directly used; data is fetched via receive_data_packet.)

    Returns:
        The processed data as a dictionary if the signature validation passes.
        Otherwise, it returns None.

    Raises:
        ConnectionError: If the packet from the device cannot be received
                         successfully (as raised by receive_data_packet).
    """
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(device_id)

    # Authenticate the packet
    if expected_signature is None or expected_signature != packet_signature:
        # Device is not trusted or signature does not match
        return None
    else:
        # Signature is valid, proceed to get the data.
        # receive_data_packet might raise ConnectionError, which will propagate.
        actual_data = receive_data_packet(device_id)
        return actual_data
