# Global trusted signatures
TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str):
    """
    Simulates receiving a data packet from an IoT device.
    Raises ConnectionError if the device_id is not recognized.
    """
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        # In a real scenario, this would involve network communication
        # and return the actual packet data.
        return {"temperature": 22.5, "humidity": 60}  # Example data
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

def process_packet(device_id: str, packet_signature: str, data_packet: dict):
    """
    Validates and processes data packets from IoT devices.

    Each packet is accompanied by a cryptographic signature that must be
    authenticated before handling the data.

    Args:
        device_id: A unique identifier for the IoT device sending the data.
        packet_signature: The cryptographic signature sent along with the packet.
        data_packet: The content of the data packet to be processed.

    Returns:
        The processed data as a dictionary if the signature validation passes.
        Otherwise, it returns None.

    Raises:
        ConnectionError: If the packet from the device cannot be received
                         successfully (simulated by receive_data_packet).
    """
    # Attempt to "receive" the packet. This also implicitly checks if the device_id
    # is known to the system, as receive_data_packet will raise ConnectionError
    # if device_id is not in TRUSTED_DEVICE_SIGNATURES.
    # The actual data returned by receive_data_packet is not used in this function,
    # as per the requirement that process_packet takes data_packet as an argument.
    receive_data_packet(device_id)

    # If receive_data_packet succeeded, it implies device_id is in TRUSTED_DEVICE_SIGNATURES.
    # Now, verify the provided packet_signature.
    expected_signature = TRUSTED_DEVICE_SIGNATURES[device_id]

    if expected_signature == packet_signature:
        return data_packet
    else:
        # Signature mismatch.
        return None
