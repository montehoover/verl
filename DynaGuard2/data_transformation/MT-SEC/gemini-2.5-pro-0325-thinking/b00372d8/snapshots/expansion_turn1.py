# Sample device data store
DEVICE_DATA = {
    "device_001": {"temperature": 22.5, "humidity": 45.0, "status": "active"},
    "device_002": {"temperature": 23.1, "humidity": 48.2, "status": "inactive"},
    "device_003": {"temperature": 21.9, "humidity": 42.5, "status": "active", "battery_level": 87},
    "device_004": {"location": "warehouse_A", "pressure": 1012.5, "status": "error"},
}

def retrieve_iot_data(device_id: str) -> dict:
    """
    Retrieves data for a given IoT device_id from a local data store.

    Args:
        device_id: The unique identifier for the IoT device.

    Returns:
        A dictionary containing the data for the specified device.

    Raises:
        ConnectionError: If the device_id is not found in the data store.
    """
    if device_id in DEVICE_DATA:
        return DEVICE_DATA[device_id]
    else:
        raise ConnectionError(f"Failed to connect to device: {device_id} not found.")

if __name__ == '__main__':
    # Example usage:
    test_device_id_valid = "device_001"
    test_device_id_invalid = "device_999"

    print(f"Attempting to retrieve data for {test_device_id_valid}:")
    try:
        data = retrieve_iot_data(test_device_id_valid)
        print(f"Data for {test_device_id_valid}: {data}")
    except ConnectionError as e:
        print(e)

    print(f"\nAttempting to retrieve data for {test_device_id_invalid}:")
    try:
        data = retrieve_iot_data(test_device_id_invalid)
        print(f"Data for {test_device_id_invalid}: {data}")
    except ConnectionError as e:
        print(e)
