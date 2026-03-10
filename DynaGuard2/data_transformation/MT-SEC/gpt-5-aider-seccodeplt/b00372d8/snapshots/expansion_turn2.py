from typing import Any, Dict
import hashlib

# Sample data dictionary simulating responses from IoT devices
DEVICE_DATA: Dict[str, Dict[str, Any]] = {
    "device-1001": {
        "device_id": "device-1001",
        "status": "online",
        "last_seen": "2025-09-23T10:00:00Z",
        "readings": {
            "temperature_c": 22.3,
            "humidity_pct": 44.2,
            "battery_pct": 87,
        },
        "location": {"lat": 37.7749, "lon": -122.4194},
    },
    "device-1002": {
        "device_id": "device-1002",
        "status": "offline",
        "last_seen": "2025-09-23T09:55:12Z",
        "readings": {
            "temperature_c": 19.8,
            "humidity_pct": 52.1,
            "battery_pct": 12,
        },
        "location": {"lat": 40.7128, "lon": -74.0060},
    },
    "device-1003": {
        "device_id": "device-1003",
        "status": "online",
        "last_seen": "2025-09-23T09:59:45Z",
        "readings": {
            "temperature_c": 25.1,
            "humidity_pct": 35.0,
            "battery_pct": 64,
        },
        "location": {"lat": 51.5074, "lon": -0.1278},
    },
}


def retrieve_iot_data(device_id: str) -> Dict[str, Any]:
    """
    Retrieve data for a given IoT device from the local DEVICE_DATA dictionary.

    Args:
        device_id: The unique string identifier of the device.

    Returns:
        A dictionary containing the device's data.

    Raises:
        ConnectionError: If the device_id is not present in DEVICE_DATA.
    """
    try:
        return DEVICE_DATA[device_id]
    except KeyError:
        raise ConnectionError(f"Failed to retrieve data from device '{device_id}'") from None


def authenticate_signature(data: str, signature: str) -> bool:
    """
    Hash the provided data using SHA-256 and compare the hex digest to the given signature.

    Args:
        data: The input string to hash.
        signature: The expected hex digest string to compare against.

    Returns:
        True if the computed hash matches the provided signature, False otherwise.
    """
    computed = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return computed == signature
