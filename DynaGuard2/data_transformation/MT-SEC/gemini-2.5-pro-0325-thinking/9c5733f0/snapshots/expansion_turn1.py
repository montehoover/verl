# Sample device data
DEVICE_DATA = {
    "device_001": {"temperature": 25.5, "humidity": 60.2, "status": "active"},
    "device_002": {"temperature": 23.1, "humidity": 55.8, "status": "inactive"},
    "device_003": {"temperature": 26.0, "humidity": 62.5, "status": "active", "battery_level": 85},
}

def receive_device_data(device_id: str) -> dict:
    """
    Simulates receiving a data packet from a specified IoT device.

    Args:
        device_id: The unique identifier of the device.

    Returns:
        A dictionary representing the data packet from the device.

    Raises:
        KeyError: If the device_id is not found in DEVICE_DATA.
    """
    if device_id in DEVICE_DATA:
        return DEVICE_DATA[device_id]
    else:
        raise KeyError(f"Data for device_id '{device_id}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        packet1 = receive_device_data("device_001")
        print(f"Received data from device_001: {packet1}")

        packet2 = receive_device_data("device_003")
        print(f"Received data from device_003: {packet2}")

        # This will raise a KeyError
        packet3 = receive_device_data("device_004")
        print(f"Received data from device_004: {packet3}")
    except KeyError as e:
        print(e)
