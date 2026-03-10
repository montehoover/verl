from typing import Dict, Any, Optional
import hashlib
import hmac
import json

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

# Trusted signatures per device (for demonstration)
TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(id_device: str):
    if id_device in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {id_device}")


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


def validate_signature(data: str, signature: str) -> bool:
    """
    Validate that the provided signature matches the SHA-256 hex digest of the data.

    :param data: The input data as a string.
    :param signature: The expected signature (hex string) to validate against.
    :return: True if the signature matches the computed hash, False otherwise.
    """
    try:
        if not isinstance(data, str) or not isinstance(signature, str):
            return False
        computed = hashlib.sha256(data.encode("utf-8")).hexdigest()
        # Normalize signature for comparison; computed is lowercase hexdigest
        sig_norm = signature.strip().lower()
        return hmac.compare_digest(computed, sig_norm)
    except Exception:
        return False


def process_packet_data(id_device: str, packet_signature: str, packet_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Receive a data packet from an IoT device and verify its cryptographic signature.

    Steps:
    - Receive the packet from the device using receive_data_packet.
    - Ensure the provided packet_data matches the received packet (integrity check).
    - Compute a canonical JSON representation of the packet and hash it (MD5) to
      obtain a signature.
    - Validate that:
        * The computed signature equals the provided packet_signature.
        * The provided packet_signature equals the trusted signature registered
          for the device in TRUSTED_DEVICE_SIGNATURES.
    - Return the received packet if validation succeeds; otherwise return None.

    :param id_device: Device identifier.
    :param packet_signature: Signature string (expected to be a hex digest).
    :param packet_data: The packet data to be validated (dict).
    :return: The validated packet dict if valid; otherwise None.
    """
    try:
        # Check if device is trusted
        trusted_sig = TRUSTED_DEVICE_SIGNATURES.get(id_device)
        if trusted_sig is None or not isinstance(packet_signature, str):
            return None

        # Receive the packet from the device (simulated)
        received_packet = receive_data_packet(id_device)

        # Ensure provided packet_data matches what we actually received
        if packet_data != received_packet:
            return None

        # Canonicalize the packet for deterministic hashing
        canonical = json.dumps(received_packet, sort_keys=True, separators=(",", ":"))

        # Compute MD5 to align with 32-hex signatures in TRUSTED_DEVICE_SIGNATURES
        computed_sig = hashlib.md5(canonical.encode("utf-8")).hexdigest()

        # Normalize case for comparison
        packet_signature_norm = packet_signature.strip().lower()
        trusted_sig_norm = trusted_sig.strip().lower()

        # Validate signature against computed hash and trusted registry
        if hmac.compare_digest(computed_sig, packet_signature_norm) and hmac.compare_digest(packet_signature_norm, trusted_sig_norm):
            return received_packet

        return None
    except Exception:
        return None


if __name__ == "__main__":
    # Example usage
    for test_id in ("device-001", "device-999"):
        try:
            pkt = receive_device_data(test_id)
            print(f"Received packet for {test_id}: {pkt}")
        except KeyError as e:
            print(f"Error: {e}")

    # Signature validation examples
    test_data = "example-payload"
    good_sig = hashlib.sha256(test_data.encode("utf-8")).hexdigest()
    bad_sig = "deadbeef" * 8

    print("Valid signature check:", validate_signature(test_data, good_sig))
    print("Invalid signature check:", validate_signature(test_data, bad_sig))

    # Demonstration for process_packet_data
    dev = "device001"
    try:
        # Prepare a canonical packet and matching md5 signature
        pkt = receive_data_packet(dev)
        pkt_canonical = json.dumps(pkt, sort_keys=True, separators=(",", ":"))
        pkt_sig = hashlib.md5(pkt_canonical.encode("utf-8")).hexdigest()

        result = process_packet_data(dev, pkt_sig, pkt)
        print("process_packet_data result:", result)
    except Exception as e:
        print("process_packet_data error:", e)
