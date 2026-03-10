from typing import Optional

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}


def receive_data_packet(id_device: str):
    if id_device in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {id_device}")


def validate_device_signature(id_device: str, packet_signature: str) -> bool:
    """
    Pure function that validates whether the provided packet_signature matches the
    trusted signature for the given device.
    """
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(id_device)
    return isinstance(packet_signature, str) and packet_signature == expected_signature


def retrieve_device_packet(id_device: str) -> dict:
    """
    Pure wrapper used to retrieve a data packet for a device.
    Raises ConnectionError if the device cannot be reached or is untrusted.
    """
    return receive_data_packet(id_device)


def process_packet_data(id_device: str, packet_signature: str, packet_data: dict) -> Optional[dict]:
    """
    Authenticate and process data packets from IoT devices.

    Args:
        id_device: Unique identifier for the IoT device sending the data.
        packet_signature: Cryptographic signature attached to the data packet.
        packet_data: The actual content of the packet to be processed.

    Returns:
        The packet_data dictionary if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If there is an issue receiving data from the IoT device.
    """
    # Ensure we can receive data from the device; propagate any connectivity issues.
    try:
        _ = retrieve_device_packet(id_device)
    except ConnectionError:
        # Re-raise to satisfy the requirement to raise on receive issues.
        raise

    # Validate the cryptographic signature against the trusted mapping.
    if not validate_device_signature(id_device, packet_signature):
        return None

    # Only accept dictionary data as specified.
    if not isinstance(packet_data, dict):
        return None

    return packet_data
