import random
import time
import hashlib
from typing import Dict, Any

# Sample dictionary of trusted device IDs for demonstration
TRUSTED_DEVICES: Dict[str, Dict[str, Any]] = {
    "device-001": {"name": "EnvSensor A", "location": "Warehouse A", "model": "DHT22"},
    "device-002": {"name": "EnvSensor B", "location": "Warehouse B", "model": "BME280"},
    "device-003": {"name": "EnvSensor C", "location": "Cold Storage", "model": "SHT31"},
}


def receive_data_packet(device_id: str) -> Dict[str, Any]:
    """
    Simulate receiving a data packet from an IoT device.

    Args:
        device_id: The unique identifier of the IoT device.

    Returns:
        A dictionary containing sample telemetry data.

    Raises:
        ConnectionError: If the device_id is not recognized.
    """
    if device_id not in TRUSTED_DEVICES:
        raise ConnectionError(f"Unrecognized device_id: {device_id}")

    # Simulate sample telemetry values
    temperature_c = round(random.uniform(18.0, 30.0), 1)
    humidity_percent = round(random.uniform(30.0, 70.0), 1)

    return {
        "device_id": device_id,
        "temperature_c": temperature_c,
        "humidity_percent": humidity_percent,
        "received_at": time.time(),
        "device_info": TRUSTED_DEVICES[device_id],
    }


def verify_packet_signature(data: str, signature: str) -> bool:
    """
    Verify a packet's signature by comparing it to the SHA-256 hash of the data.

    Args:
        data: The data string to hash.
        signature: The expected hex-encoded SHA-256 digest.

    Returns:
        True if the computed digest matches the provided signature, False otherwise.
        Returns False on any error or unsupported operation.
    """
    try:
        if data is None or signature is None:
            return False
        computed = hashlib.sha256(data.encode("utf-8")).hexdigest()
        return computed == signature.strip().lower()
    except Exception:
        return False
