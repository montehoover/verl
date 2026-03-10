from typing import Optional, Dict, Any
import hmac

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


def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: dict) -> Optional[dict]:
    """
    Validates and processes data packets from IoT devices.

    - Authenticates the cryptographic signature for the given device_id.
    - Attempts to receive a data packet from the device. If receiving fails, raises ConnectionError.
    - On successful validation and receipt, returns processed data as a dictionary.
      Returns None if signature validation fails.

    Args:
        device_id: Unique identifier for the IoT device.
        crypt_sig: Cryptographic signature accompanying the packet.
        pkt_data: The content of the data packet to be processed.

    Returns:
        A dictionary of processed data if signature validation passes; otherwise None.

    Raises:
        ConnectionError: If the packet from the device cannot be received successfully.
    """
    # Basic type validation
    if not isinstance(device_id, str) or not isinstance(crypt_sig, str) or not isinstance(pkt_data, dict):
        return None

    # Validate signature for the device
    expected_sig = TRUSTED_DEVICE_SIGNATURES.get(device_id)
    if expected_sig is None:
        # Unknown device
        return None

    if not hmac.compare_digest(crypt_sig, expected_sig):
        # Signature mismatch
        return None

    # Attempt to receive the packet; may raise ConnectionError as specified
    received_packet = receive_data_packet(device_id)

    if not isinstance(received_packet, dict):
        # Defensive check: treat non-dict as a receive failure scenario
        raise ConnectionError(f"Malformed packet received from device {device_id}")

    # Process the data: merge received packet with provided pkt_data.
    # Values in pkt_data take precedence over received_packet on key conflicts.
    processed = {**received_packet, **pkt_data}

    return processed
