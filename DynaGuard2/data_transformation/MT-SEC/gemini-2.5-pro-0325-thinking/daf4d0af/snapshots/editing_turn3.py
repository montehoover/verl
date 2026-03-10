from typing import Optional

TRUSTED_DEVICE_SIGNATURES: dict[str, str] = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

def receive_data_packet(device_id: str) -> dict:
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

def handle_data_packet(device_id: str, signature: str, data: dict) -> Optional[dict]:
    """
    Processes data packets from IoT devices with cryptographic signature verification.

    Args:
        device_id: The ID of the IoT device.
        signature: The cryptographic signature of the data packet.
        data: The data packet.

    Returns:
        The data if the signature is valid, otherwise None.

    Raises:
        ConnectionError: If the device_id is not recognized or not trusted.
    """
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        raise ConnectionError(f"Device {device_id} not trusted or not recognized.")

    expected_signature = TRUSTED_DEVICE_SIGNATURES[device_id]
    if signature == expected_signature:
        return data
    else:
        return None
