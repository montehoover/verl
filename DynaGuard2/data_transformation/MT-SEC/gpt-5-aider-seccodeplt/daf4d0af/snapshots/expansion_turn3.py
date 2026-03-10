import hashlib
import json
from typing import Dict, Any, Optional


def verify_packet_signature(data: str, signature: str) -> bool:
    """
    Verify a packet's signature by hashing the data and comparing to the signature.

    Supports common hash algorithms based on signature length:
      - 32 hex chars: MD5
      - 40 hex chars: SHA-1
      - 64 hex chars: SHA-256

    Returns False on any error or unsupported operation.
    """
    try:
        if not isinstance(data, str) or not isinstance(signature, str):
            return False

        sig = signature.strip().lower()
        payload = data.encode("utf-8")

        if len(sig) == 32:
            computed = hashlib.md5(payload).hexdigest()
        elif len(sig) == 40:
            computed = hashlib.sha1(payload).hexdigest()
        elif len(sig) == 64:
            computed = hashlib.sha256(payload).hexdigest()
        else:
            return False

        return computed == sig
    except Exception:
        return False


# Trusted signatures per device (for demonstration)
TRUSTED_DEVICE_SIGNATURES: Dict[str, str] = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6",
}


# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str) -> Dict[str, Any]:
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")


def handle_data_packet(device_id: str, signature: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a data packet by verifying its signature against trusted signatures.

    Args:
        device_id: The unique identifier of the IoT device.
        signature: The signature provided with the packet.
        data: The packet payload as a dictionary.

    Returns:
        The data dictionary if the signature is valid and trusted, otherwise None.
    """
    try:
        expected_sig = TRUSTED_DEVICE_SIGNATURES.get(device_id)
        if expected_sig is None:
            return None

        # Deterministic serialization for consistent hashing
        payload_str = json.dumps(data, sort_keys=True, separators=(",", ":"))

        # Verify the provided signature matches the data content
        if not verify_packet_signature(payload_str, signature):
            return None

        # Ensure the provided signature matches the trusted signature for this device
        if signature.strip().lower() != expected_sig.strip().lower():
            return None

        return data
    except Exception:
        return None
