import hashlib
import json

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3", # Example signature value
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"  # Example signature value
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str) -> dict: # Return type is dict or raises ConnectionError
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        # This is a fixed simulation based on the setup code
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

def validate_signature(data: str, signature: str) -> bool:
    """
    Validates the signature of the given data.

    Args:
        data: The data string to validate.
        signature: The expected signature (hex digest of SHA256).

    Returns:
        True if the signature matches the computed hash of the data, False otherwise.
        Returns False for any unsupported inputs (e.g., non-string types).
    """
    if not isinstance(data, str) or not isinstance(signature, str):
        return False  # Unsupported input types

    try:
        # Create a new SHA256 hash object
        hasher = hashlib.sha256()
        # Update the hasher with the bytes of the data string
        hasher.update(data.encode('utf-8'))
        # Get the hexadecimal representation of the hash
        computed_signature = hasher.hexdigest()
        # Compare the computed signature with the provided signature
        return computed_signature == signature
    except Exception:
        # Catch any other unexpected errors during hashing or comparison
        return False

def process_packet(device_id: str, packet_signature: str, data_packet: dict) -> dict:
    """
    Validates and processes a data packet from an IoT device.

    Args:
        device_id: The ID of the device.
        packet_signature: The signature received with the packet.
        data_packet: The data packet (dictionary).

    Returns:
        The processed data packet if validation is successful.

    Raises:
        ValueError: If the device ID is untrusted, data packet is not serializable,
                    or the packet signature is invalid.
    """
    if device_id not in TRUSTED_DEVICE_SIGNATURES:
        raise ValueError(f"Untrusted device ID: {device_id}")

    try:
        # Serialize data_packet to a canonical JSON string for consistent hashing.
        # sort_keys ensures order, separators ensures compact representation.
        data_string = json.dumps(data_packet, sort_keys=True, separators=(',', ':'))
    except TypeError:
        # This can happen if data_packet contains non-JSON-serializable types like sets
        raise ValueError("Data packet is not JSON serializable")

    if validate_signature(data_string, packet_signature):
        # "Processing" currently means returning the validated packet.
        # More complex processing logic could be added here in the future.
        return data_packet
    else:
        raise ValueError("Invalid packet signature")

if __name__ == '__main__':
    # Example usage for the new receive_data_packet:
    test_device_id_trusted = "device001" # Updated device ID based on TRUSTED_DEVICE_SIGNATURES

    print(f"Attempting to retrieve data for trusted device {test_device_id_trusted}:")
    try:
        packet = receive_data_packet(test_device_id_trusted)
        # Expected: {"temperature": 22.5, "humidity": 60} from the new simulated function
        print(f"Data packet for {test_device_id_trusted}: {packet}")
    except ConnectionError as e:
        print(e)

    # Example usage for validate_signature (structure remains, context might change slightly):
    sample_data = "This is some important data from the IoT device."
    # Pre-calculate a signature for the sample_data
    # In a real scenario, the device would generate this signature with a secret key or method
    # For this example, we'll just compute it directly
    hasher_example = hashlib.sha256()
    hasher_example.update(sample_data.encode('utf-8'))
    correct_signature = hasher_example.hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"

    print(f"\nValidating correct signature for data: '{sample_data}'")
    is_valid = validate_signature(sample_data, correct_signature)
    print(f"Signature validation result: {is_valid}") # Expected: True

    print(f"\nValidating incorrect signature for data: '{sample_data}'")
    is_valid = validate_signature(sample_data, incorrect_signature)
    print(f"Signature validation result: {is_valid}") # Expected: False

    print(f"\nValidating with non-string data:")
    is_valid = validate_signature(12345, correct_signature) # type: ignore
    print(f"Signature validation result for non-string data: {is_valid}") # Expected: False

    print(f"\nValidating with non-string signature:")
    is_valid = validate_signature(sample_data, None) # type: ignore
    print(f"Signature validation result for non-string signature: {is_valid}") # Expected: False

    # Test for untrusted device with the new receive_data_packet
    test_device_id_untrusted = "device999" # This ID is not in TRUSTED_DEVICE_SIGNATURES
    print(f"\nAttempting to retrieve data for untrusted device {test_device_id_untrusted}:")
    try:
        packet = receive_data_packet(test_device_id_untrusted)
        print(f"Data packet for {test_device_id_untrusted}: {packet}")
    except ConnectionError as e:
        print(e) # Expected: ConnectionError: Failed to receive packet from device device999

    # Example usage for process_packet:
    print("\n--- process_packet Examples ---")

    # Case 1: Trusted device, valid signature and data
    proc_device1_id = "device001" # Must be in TRUSTED_DEVICE_SIGNATURES
    proc_device1_data_packet = {"sensor": "A", "value": 42, "timestamp": "2025-01-01T12:00:00Z"}
    # Create a canonical string representation for hashing
    proc_device1_data_string = json.dumps(proc_device1_data_packet, sort_keys=True, separators=(',', ':'))
    
    hasher_proc_dev1 = hashlib.sha256()
    hasher_proc_dev1.update(proc_device1_data_string.encode('utf-8'))
    proc_device1_correct_signature = hasher_proc_dev1.hexdigest()

    print(f"\nProcessing packet from trusted device {proc_device1_id} with correct signature:")
    try:
        processed_packet = process_packet(proc_device1_id, proc_device1_correct_signature, proc_device1_data_packet)
        print(f"Processed packet: {processed_packet}") # Expected: the data_packet itself
    except ValueError as e:
        print(e)

    # Case 2: Trusted device, invalid signature
    proc_device1_incorrect_signature = "invalidpacketsignature123abc" # Not a valid SHA256 for the data
    print(f"\nProcessing packet from trusted device {proc_device1_id} with incorrect signature:")
    try:
        processed_packet = process_packet(proc_device1_id, proc_device1_incorrect_signature, proc_device1_data_packet)
        print(f"Processed packet: {processed_packet}")
    except ValueError as e:
        print(e) # Expected: ValueError("Invalid packet signature")

    # Case 3: Untrusted device
    proc_untrusted_device_id = "device777" # Not in TRUSTED_DEVICE_SIGNATURES
    # Data and signature for this case don't strictly matter as device ID check should fail first
    some_other_data_packet = {"data": "any_payload"}
    some_other_signature = proc_device1_correct_signature 
    print(f"\nProcessing packet from untrusted device {proc_untrusted_device_id}:")
    try:
        processed_packet = process_packet(proc_untrusted_device_id, some_other_signature, some_other_data_packet)
        print(f"Processed packet: {processed_packet}")
    except ValueError as e:
        print(e) # Expected: ValueError("Untrusted device ID: device777")

    # Case 4: Trusted device, non-serializable data packet
    # A set is not directly JSON serializable by json.dumps by default
    non_serializable_packet = {"data_payload": {1, 2, 3}, "sensor_id": "sensor_xyz"} 
    print(f"\nProcessing packet from trusted device {proc_device1_id} with non-serializable data:")
    try:
        # Using a correct-looking signature, but serialization of the packet should fail
        processed_packet = process_packet(proc_device1_id, proc_device1_correct_signature, non_serializable_packet)
        print(f"Processed packet: {processed_packet}")
    except ValueError as e:
        print(e) # Expected: ValueError("Data packet is not JSON serializable")
