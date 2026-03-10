from typing import Any, Dict, Optional
import hashlib
import json

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


TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str):
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")


def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and process an IoT data packet.

    Steps:
    - Ensure the device_id is trusted.
    - Ensure the provided cryptographic signature matches the trusted signature for the device.
    - Hash the canonical JSON representation of pkt_data and verify it matches the provided signature.
    - If all checks pass, return a processed/merged data dictionary; otherwise, return None.
    """
    # Check device trust and signature registration
    trusted_sig = TRUSTED_DEVICE_SIGNATURES.get(device_id)
    if trusted_sig is None:
        return None
    if crypt_sig != trusted_sig:
        return None

    # Canonicalize payload to a stable string for hashing and signature verification
    try:
        payload_str = json.dumps(pkt_data, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None

    # Verify that the payload's signature matches
    if not authenticate_signature(payload_str, crypt_sig):
        return None

    # Optionally receive a baseline packet for the device and merge with provided data.
    # Provided pkt_data values take precedence.
    try:
        baseline = receive_data_packet(device_id)
        if isinstance(baseline, dict):
            processed = dict(baseline)
            processed.update(pkt_data)
        else:
            processed = pkt_data
    except ConnectionError:
        processed = pkt_data

    return processed
