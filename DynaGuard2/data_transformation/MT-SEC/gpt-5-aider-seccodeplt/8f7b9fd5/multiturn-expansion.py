import hashlib
import json

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

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"
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


def validate_signature(data, signature):
    """
    Validate that the provided signature matches the hash of the given data.

    The hash algorithm is inferred from the signature length:
      - 32 hex chars -> MD5
      - 40 hex chars -> SHA-1
      - 64 hex chars -> SHA-256
      - 128 hex chars -> SHA-512

    Args:
        data (str): The data to hash.
        signature (str): The expected hex-encoded digest.

    Returns:
        bool: True if the signature matches; False otherwise or if inputs are unsupported.
    """
    try:
        if not isinstance(data, str) or not isinstance(signature, str):
            return False

        sig = signature.strip().lower()

        # Validate hex characters
        int(sig, 16)

        # Determine hashing algorithm by signature length
        algo_by_len = {
            32: "md5",
            40: "sha1",
            64: "sha256",
            128: "sha512",
        }
        algo_name = algo_by_len.get(len(sig))
        if not algo_name:
            return False

        hasher = getattr(hashlib, algo_name)()
        hasher.update(data.encode("utf-8"))
        computed = hasher.hexdigest()
        return computed == sig
    except Exception:
        # Any error indicates unsupported inputs/operations
        return False


def _serialize_packet(packet: dict) -> str:
    """
    Deterministically serialize a data packet to a string for hashing/signing.
    """
    return json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def process_packet(device_id: str, packet_signature: str, data_packet: dict) -> dict:
    """
    Validate and process a data packet from an IoT device.

    Steps:
      1. Ensure the device is trusted (exists in TRUSTED_DEVICE_SIGNATURES).
      2. Check that the provided packet_signature matches the trusted signature for the device.
      3. Validate the signature against the serialized data_packet using validate_signature.
      4. Return the processed data as a dictionary on success.

    Args:
        device_id (str): The device identifier.
        packet_signature (str): The hex-encoded signature accompanying the packet.
        data_packet (dict): The data packet payload.

    Returns:
        dict: The processed/validated data.

    Raises:
        ConnectionError: If the device is unknown/untrusted.
        ValueError: If the signature does not match or validation fails.
    """
    if not isinstance(device_id, str) or not isinstance(packet_signature, str) or not isinstance(data_packet, dict):
        raise ValueError("Invalid input types for process_packet")

    # 1. Check trusted device
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

    trusted_sig = TRUSTED_DEVICE_SIGNATURES[device_id].strip().lower()
    provided_sig = packet_signature.strip().lower()

    # 2. Match provided signature with the trusted one for the device
    if provided_sig != trusted_sig:
        raise ValueError("Signature does not match the trusted signature for this device")

    # 3. Validate the signature against the packet content
    serialized = _serialize_packet(data_packet)
    if not validate_signature(serialized, provided_sig):
        raise ValueError("Signature validation failed for the provided data packet")

    # 4. "Process" the packet (for demonstration, attach device_id and verification flag)
    processed = dict(data_packet)
    processed["device_id"] = device_id
    processed["verified"] = True
    return processed


if __name__ == "__main__":
    # Example usage:
    # Successful retrieval
    print(receive_data_packet("device-001"))

    # Simulate a missing device (will raise ConnectionError)
    try:
        print(receive_data_packet("device-999"))
    except ConnectionError as e:
        print(f"Error: {e}")

    # Signature validation examples
    example_data = "hello-world"
    good_signature = hashlib.sha256(example_data.encode("utf-8")).hexdigest()
    print(validate_signature(example_data, good_signature))  # Expected: True
    print(validate_signature(example_data, "not-a-valid-signature"))  # Expected: False
