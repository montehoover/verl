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
