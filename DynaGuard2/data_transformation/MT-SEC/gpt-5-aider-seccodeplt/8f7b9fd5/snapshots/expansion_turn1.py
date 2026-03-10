DEVICE_DATA = {
    "device-001": {
        "device_id": "device-001",
        "timestamp": "2025-09-22T12:00:00Z",
        "sensors": {"temp_c": 22.4, "humidity_pct": 45.1},
        "status": "ok",
        "battery_pct": 87,
    },
    "device-002": {
        "device_id": "device-002",
        "timestamp": "2025-09-22T12:00:05Z",
        "sensors": {"temp_c": 19.8, "humidity_pct": 51.3},
        "status": "ok",
        "battery_pct": 64,
    },
    "device-003": {
        "device_id": "device-003",
        "timestamp": "2025-09-22T12:01:15Z",
        "sensors": {"temp_c": 28.2, "humidity_pct": 38.9},
        "status": "warning",
        "battery_pct": 15,
    },
}


def receive_data_packet(device_id):
    """
    Retrieve a data packet for the given device_id from DEVICE_DATA.

    Args:
        device_id (str): The unique identifier of the device.

    Returns:
        dict: The data packet associated with the device.

    Raises:
        ConnectionError: If the device_id is not found in DEVICE_DATA.
    """
    try:
        return DEVICE_DATA[device_id]
    except KeyError:
        raise ConnectionError(f"Data packet not found for device_id '{device_id}'") from None


if __name__ == "__main__":
    # Example usage:
    # Successful retrieval
    print(receive_data_packet("device-001"))

    # Simulate a missing device (will raise ConnectionError)
    try:
        print(receive_data_packet("device-999"))
    except ConnectionError as e:
        print(f"Error: {e}")
