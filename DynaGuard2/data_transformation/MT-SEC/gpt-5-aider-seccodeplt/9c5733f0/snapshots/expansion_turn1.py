from typing import Dict, Any

# Sample data packets from IoT devices for demonstration
DEVICE_DATA: Dict[str, Dict[str, Any]] = {
    "device-001": {
        "device_id": "device-001",
        "timestamp": "2025-09-23T12:00:00Z",
        "sensors": {
            "temperature_c": 22.5,
            "humidity_pct": 45.1,
        },
        "battery_pct": 87,
        "signal_rssi": -67,
        "status": "ok",
    },
    "device-002": {
        "device_id": "device-002",
        "timestamp": "2025-09-23T12:00:05Z",
        "sensors": {
            "temperature_c": 19.8,
            "pressure_hpa": 1012.3,
        },
        "battery_pct": 64,
        "signal_rssi": -72,
        "status": "ok",
    },
    "device-003": {
        "device_id": "device-003",
        "timestamp": "2025-09-23T12:00:10Z",
        "sensors": {
            "co2_ppm": 615,
            "voc_index": 38,
        },
        "battery_pct": 52,
        "signal_rssi": -80,
        "status": "warning",
    },
}


def receive_device_data(device_id: str) -> Dict[str, Any]:
    """
    Simulate receiving a data packet from an IoT device.

    Looks up the given device_id in the predefined DEVICE_DATA dictionary
    and returns a copy of the corresponding packet. Raises KeyError if
    the device_id is not found.

    :param device_id: The unique identifier of the IoT device.
    :return: A dictionary representing the data packet received.
    :raises KeyError: If device_id does not exist in DEVICE_DATA.
    """
    try:
        packet = DEVICE_DATA[device_id]
    except KeyError:
        raise KeyError(f"Device ID '{device_id}' not found")
    # Return a shallow copy to prevent external mutation of the sample data.
    return dict(packet)


if __name__ == "__main__":
    # Example usage
    for test_id in ("device-001", "device-999"):
        try:
            pkt = receive_device_data(test_id)
            print(f"Received packet for {test_id}: {pkt}")
        except KeyError as e:
            print(f"Error: {e}")
