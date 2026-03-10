from typing import Optional, Set

# Predefined list of known device IDs.
KNOWN_DEVICES: Set[str] = {
    "device-001",
    "device-002",
    "device-003",
}

def process_iot_data(device_id: str, data_packet: dict) -> Optional[dict]:
    """
    Process data packets from IoT devices.

    Parameters:
        device_id (str): The ID of the IoT device.
        data_packet (dict): The data packet sent by the device.

    Returns:
        dict | None: The data packet if the device is recognized; otherwise None.
    """
    if device_id not in KNOWN_DEVICES:
        return None
    return data_packet


# Trusted device signatures for validating packets.
TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

def receive_data_packet(device_id: str):
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

def process_packet(device_id: str, packet_signature: str, data_packet: dict) -> Optional[dict]:
    """
    Validate and process data packets from IoT devices using cryptographic signatures.

    Parameters:
        device_id (str): The ID of the IoT device.
        packet_signature (str): The cryptographic signature supplied with the packet.
        data_packet (dict): The data packet sent by the device.

    Returns:
        dict | None: The data packet if signature validation passes; otherwise None.
    """
    expected_signature = TRUSTED_DEVICE_SIGNATURES.get(device_id)
    if expected_signature is None:
        return None
    if packet_signature != expected_signature:
        return None
    return data_packet
