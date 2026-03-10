import random
import hashlib
import json

TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3", # Example signature, not used by verify_packet_signature
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"  # Example signature, not used by verify_packet_signature
}

# Simulated function to receive a data packet from the IoT device
def receive_data_packet(device_id: str) -> dict:
    """
    Simulates receiving a data packet from an IoT device.
    Checks if the device_id is in TRUSTED_DEVICE_SIGNATURES.

    Args:
        device_id: The ID of the device.

    Returns:
        A dictionary with sample data if device is trusted.

    Raises:
        ConnectionError: If the device_id is not in TRUSTED_DEVICE_SIGNATURES.
    """
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        return {"temperature": 22.5, "humidity": 60}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}")

def verify_packet_signature(data: str, signature: str) -> bool:
    """
    Verifies the signature of a data packet.

    Args:
        data: The data string to verify.
        signature: The expected signature (hex digest of SHA256).

    Returns:
        True if the signature matches the data's hash, False otherwise.
    """
    try:
        # Ensure data is encoded to bytes before hashing
        data_bytes = data.encode('utf-8')
        computed_hash = hashlib.sha256(data_bytes).hexdigest()
        return computed_hash == signature
    except (AttributeError, TypeError): # AttributeError for potential issues with hashlib, TypeError for encoding
        # Handle cases where hashing might not be supported or data is not string
        return False

def handle_data_packet(device_id: str, signature: str, data: dict) -> dict | None:
    """
    Processes a data packet: verifies its signature and returns data if valid.

    Args:
        device_id: The ID of the device sending the data. (Currently not used in core
                   verification logic but available for future use e.g. logging, key lookup)
        signature: The signature of the data packet (hex digest of SHA256).
        data: The data packet content as a dictionary.

    Returns:
        The data dictionary if the signature is valid, otherwise None.
    """
    if not isinstance(data, dict):
        # Data must be a dictionary to be processed
        return None
    try:
        # Serialize the dictionary to a canonical JSON string for consistent hashing.
        # sort_keys ensures order, separators removes whitespace for compactness.
        data_string = json.dumps(data, sort_keys=True, separators=(',', ':'))
    except TypeError:
        # Data might contain non-JSON-serializable elements
        return None

    if verify_packet_signature(data_string, signature):
        return data
    else:
        return None

if __name__ == '__main__':
    print("--- Testing receive_data_packet ---")
    test_device_trusted = "device001"
    test_device_untrusted = "device999"

    try:
        packet = receive_data_packet(test_device_trusted)
        print(f"Received data from trusted device {test_device_trusted}: {packet}")
    except ConnectionError as e:
        print(e)

    try:
        packet = receive_data_packet(test_device_untrusted)
        print(f"Attempting to receive data from untrusted device {test_device_untrusted}:")
    except ConnectionError as e:
        print(e)

    print("\n--- Testing verify_packet_signature (already tested but good to keep) ---")
    sample_data_good = "This is some important data."
    correct_signature = hashlib.sha256(sample_data_good.encode('utf-8')).hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"
    print(f"Verifying good data with correct signature: {verify_packet_signature(sample_data_good, correct_signature)}")
    print(f"Verifying good data with incorrect signature: {verify_packet_signature(sample_data_good, incorrect_signature)}")

    print("\n--- Testing handle_data_packet ---")
    # Sample data packet content
    sample_packet_content = {"temperature": 25.1, "humidity": 55.5, "sensor_id": "XYZ789"}

    # Convert dict to canonical JSON string for signing (sender would do this)
    sample_packet_content_str = json.dumps(sample_packet_content, sort_keys=True, separators=(',', ':'))
    
    # Generate the correct signature for the sample packet content (sender would do this)
    correct_packet_signature = hashlib.sha256(sample_packet_content_str.encode('utf-8')).hexdigest()
    
    # Create an incorrect signature for testing
    incorrect_packet_signature = "0000" + correct_packet_signature[4:] # Tampered

    device_for_handling = "device001" # A device ID from TRUSTED_DEVICE_SIGNATURES

    # Test 1: Handle packet with correct signature
    handled_data_correct = handle_data_packet(device_for_handling, correct_packet_signature, sample_packet_content)
    print(f"Handling packet from '{device_for_handling}' with correct signature: {handled_data_correct}")

    # Test 2: Handle packet with incorrect signature
    handled_data_incorrect = handle_data_packet(device_for_handling, incorrect_packet_signature, sample_packet_content)
    print(f"Handling packet from '{device_for_handling}' with incorrect signature: {handled_data_incorrect}")

    # Test 3: Handle packet with non-dict data
    handled_data_bad_type = handle_data_packet(device_for_handling, correct_packet_signature, "this is not a dict")
    print(f"Handling packet from '{device_for_handling}' with non-dict data type: {handled_data_bad_type}")

    # Test 4: Handle packet with data that cannot be JSON serialized (e.g. set)
    non_serializable_data = {"value": {1, 2, 3}} # Sets are not directly JSON serializable by default
    non_serializable_data_str = json.dumps(non_serializable_data, sort_keys=True, default=str, separators=(',',':')) # Need default=str for set
    sig_for_non_serializable = hashlib.sha256(non_serializable_data_str.encode('utf-8')).hexdigest()
    
    # The handle_data_packet will fail to serialize if default=str is not used internally.
    # Our current handle_data_packet does not use default=str, so this will test the try-except for TypeError.
    handled_data_non_serializable = handle_data_packet(device_for_handling, sig_for_non_serializable, non_serializable_data)
    print(f"Handling packet from '{device_for_handling}' with non-JSON-serializable data: {handled_data_non_serializable}")
