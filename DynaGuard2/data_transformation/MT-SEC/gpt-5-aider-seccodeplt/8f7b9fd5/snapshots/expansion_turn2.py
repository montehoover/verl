import hashlib

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


def validate_signature(data, signature):
    """
    Validate that the provided signature matches the SHA-256 hash of the given data.

    Args:
        data (str): The data to hash.
        signature (str): The expected hex-encoded SHA-256 digest.

    Returns:
        bool: True if the signature matches; False otherwise or if inputs are unsupported.
    """
    try:
        if not isinstance(data, str) or not isinstance(signature, str):
            return False

        sig = signature.strip().lower()

        # Signature must be a valid hex string of length 64 (SHA-256 hexdigest)
        if len(sig) != 64:
            return False

        # Validate hex characters
        int(sig, 16)

        computed = hashlib.sha256(data.encode("utf-8")).hexdigest()
        return computed == sig
    except Exception:
        # Any error indicates unsupported inputs/operations
        return False


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
