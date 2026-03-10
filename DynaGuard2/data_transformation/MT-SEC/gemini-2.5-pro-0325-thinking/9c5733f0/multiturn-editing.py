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

def process_packet_data(id_device: str, packet_signature: str, packet_data: dict) -> dict | None:
    """
    Authenticates a data packet using a cryptographic signature and fetches data.

    Args:
        id_device: The identifier of the device.
        packet_signature: The cryptographic signature provided by the device.
        packet_data: The data packet (currently unused in this version,
                     but kept for interface consistency or future use).

    Returns:
        The data packet (a dictionary) from the IoT device if the signature is valid.
        Returns None if the signature is invalid or the device is not recognized.

    Raises:
        ConnectionError: If there is an issue receiving data from the IoT device
                         after successful authentication.
    """
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(id_device)

    if expected_signature is None or packet_signature != expected_signature:
        # Device not in trusted list or signature mismatch
        return None

    # Signature is valid, proceed to fetch data.
    # The call to receive_data_packet might raise a ConnectionError,
    # which will propagate up from this function as per requirements.
    device_data = receive_data_packet(id_device)
    return device_data
