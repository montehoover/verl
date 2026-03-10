from typing import Optional

KNOWN_DEVICES = ["device_001", "device_002", "device_003"]

def process_iot_data(device_id: str, data_packet: dict) -> Optional[dict]:
    """
    Processes data packets from IoT devices.
    Validates if the device_id is recognized.

    Args:
        device_id: The ID of the IoT device.
        data_packet: The data packet from the device.

    Returns:
        The data packet if the device_id is recognized, otherwise None.
    """
    if device_id not in KNOWN_DEVICES:
        return None
    return data_packet
