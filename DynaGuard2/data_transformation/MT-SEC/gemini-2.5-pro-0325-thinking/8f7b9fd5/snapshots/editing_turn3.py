from typing import Optional, Dict

TRUSTED_DEVICE_SIGNATURES: Dict[str, str] = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

def receive_data_packet(device_id: str) -> Dict:
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

def process_packet(device_id: str, packet_signature: str, data_packet: dict) -> Optional[dict]:
    """
    Validates and processes data packets from IoT devices using cryptographic signatures.

    Args:
        device_id: The ID of the IoT device.
        packet_signature: The cryptographic signature of the packet.
        data_packet: The data packet from the device.

    Returns:
        The processed data as a dictionary if the signature validation passes, and None otherwise.
    """
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(device_id)
    if expected_signature and expected_signature == packet_signature:
        # In a real scenario, further processing of data_packet might happen here.
        return data_packet
    return None
