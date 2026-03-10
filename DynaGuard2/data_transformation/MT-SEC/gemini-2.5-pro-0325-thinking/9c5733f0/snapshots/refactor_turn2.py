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

def _is_signature_valid(id_device: str, packet_signature: str, trusted_signatures: dict) -> bool:
    """
    Validates the packet signature against trusted signatures.

    Args:
        id_device: The unique identifier for the IoT device.
        packet_signature: The cryptographic signature of the packet.
        trusted_signatures: A dictionary of trusted device IDs to their signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    if id_device not in trusted_signatures:
        return False
    return packet_signature == trusted_signatures[id_device]

def process_packet_data(id_device: str, packet_signature: str, packet_data: dict):
    """
    Authenticates and processes data packets from IoT devices.

    Args:
        id_device: The unique identifier for the IoT device.
        packet_signature: The cryptographic signature of the packet.
        packet_data: The content of the packet.

    Returns:
        The packet_data if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If there is an issue receiving data from the IoT device
                         (as determined by receive_data_packet).
    """
    try:
        # This call is to meet the requirement that process_packet_data
        # should raise a ConnectionError if data reception fails.
        # The receive_data_packet function simulates this.
        # The data returned by receive_data_packet() is not used here,
        # as the function is instructed to process the passed 'packet_data'.
        receive_data_packet(id_device)
    except ConnectionError:
        # Propagate the ConnectionError if receive_data_packet raises it.
        raise

    # If receive_data_packet succeeded, proceed with signature validation.
    if _is_signature_valid(id_device, packet_signature, TRUSTED_DEVICE_SIGNATURES):
        return packet_data
    else:
        return None
