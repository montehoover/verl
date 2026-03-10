# Example DEVICE_DATA dictionary
DEVICE_DATA = {
    "device_001": {"timestamp": "2025-05-24T10:00:00Z", "temperature": 25.5, "humidity": 60.1},
    "device_002": {"timestamp": "2025-05-24T10:01:00Z", "pressure": 1012.5, "light_level": 300},
    "device_003": {"timestamp": "2025-05-24T10:02:00Z", "temperature": 22.3, "co2_level": 450},
}

def receive_data_packet(device_id: str) -> dict:
    """
    Retrieves a data packet for a given device_id from the DEVICE_DATA dictionary.

    Args:
        device_id: The ID of the device.

    Returns:
        The data packet (dictionary) for the device.

    Raises:
        ConnectionError: If the device_id is not found in DEVICE_DATA.
    """
    if device_id in DEVICE_DATA:
        return DEVICE_DATA[device_id]
    else:
        raise ConnectionError(f"Data packet not found for device ID: {device_id}")

if __name__ == '__main__':
    # Example usage:
    test_device_id_exists = "device_001"
    test_device_id_not_exists = "device_999"

    print(f"Attempting to retrieve data for {test_device_id_exists}:")
    try:
        packet = receive_data_packet(test_device_id_exists)
        print(f"Data packet for {test_device_id_exists}: {packet}")
    except ConnectionError as e:
        print(e)

    print(f"\nAttempting to retrieve data for {test_device_id_not_exists}:")
    try:
        packet = receive_data_packet(test_device_id_not_exists)
        print(f"Data packet for {test_device_id_not_exists}: {packet}")
    except ConnectionError as e:
        print(e)
