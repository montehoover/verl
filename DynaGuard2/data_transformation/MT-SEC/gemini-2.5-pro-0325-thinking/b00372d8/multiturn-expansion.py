import hashlib
import json

# Sample device data store
DEVICE_DATA = {
    "device_001": {"temperature": 22.5, "humidity": 45.0, "status": "active"},
    "device_002": {"temperature": 23.1, "humidity": 48.2, "status": "inactive"},
    "device_003": {"temperature": 21.9, "humidity": 42.5, "status": "active", "battery_level": 87},
    "device_004": {"location": "warehouse_A", "pressure": 1012.5, "status": "error"},
}

# Trusted device signatures (as provided in the setup code)
TRUSTED_DEVICE_SIGNATURES = {
    "device001": "89ebf76ae45bfcf6dd7f2aa3de8b20b3",  # Note: These appear to be MD5-like.
    "device002": "1dcfa479db7aae3d40b3984cdf202de6"   # authenticate_signature uses SHA256.
}

def retrieve_iot_data(device_id: str) -> dict:
    """
    Retrieves data for a given IoT device_id from a local data store.

    Args:
        device_id: The unique identifier for the IoT device.

    Returns:
        A dictionary containing the data for the specified device.

    Raises:
        ConnectionError: If the device_id is not found in the data store.
    """
    if device_id in DEVICE_DATA:
        return DEVICE_DATA[device_id]
    else:
        raise ConnectionError(f"Failed to connect to device: {device_id} not found.")

def authenticate_signature(data: str, signature: str) -> bool:
    """
    Authenticates data by comparing its hash with a provided signature.

    Args:
        data: The data string to authenticate.
        signature: The expected hash signature (hexadecimal string).

    Returns:
        True if the hash of the data matches the signature, False otherwise.
    """
    # Create a new SHA256 hash object
    hasher = hashlib.sha256()
    # Update the hash object with the bytes of the data string
    hasher.update(data.encode('utf-8'))
    # Get the hexadecimal representation of the hash
    calculated_signature = hasher.hexdigest()
    # Compare the calculated signature with the provided signature
    return calculated_signature == signature

# Simulated function to receive a data packet from the IoT device (as provided in setup)
def receive_data_packet(device_id: str) -> dict:
    """
    Simulates receiving a data packet from an IoT device.
    Uses TRUSTED_DEVICE_SIGNATURES to determine if data can be "received".
    """
    if device_id in TRUSTED_DEVICE_SIGNATURES:
        # Simulate a generic data packet structure
        return {"temperature": 22.5, "humidity": 60.0}
    else:
        raise ConnectionError(f"Failed to receive packet from device {device_id}: Device not in trusted list.")

def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: dict) -> dict | None:
    """
    Validates and processes data packets from IoT devices.

    Args:
        device_id: The ID of the device sending the packet. (Currently unused in core logic but good for context)
        crypt_sig: The cryptographic signature provided with the packet data.
        pkt_data: The data packet as a dictionary.

    Returns:
        The processed data dictionary if the signature is verified successfully, 
        None otherwise.
    """
    # Serialize pkt_data to a canonical string format for hashing.
    # Using json.dumps with sort_keys=True and compact separators ensures consistency.
    data_string_to_verify = json.dumps(pkt_data, sort_keys=True, separators=(',', ':'))

    if authenticate_signature(data_string_to_verify, crypt_sig):
        return pkt_data
    else:
        return None

if __name__ == '__main__':
    # Example usage for retrieve_iot_data:
    test_device_id_valid = "device_001"
    test_device_id_invalid = "device_999"

    print(f"Attempting to retrieve data for {test_device_id_valid}:")
    try:
        data = retrieve_iot_data(test_device_id_valid)
        print(f"Data for {test_device_id_valid}: {data}")
    except ConnectionError as e:
        print(e)

    # --- New example usage for process_iot_packet ---
    print(f"\n--- Testing process_iot_packet ---")

    # Test Case 1: Successful processing of a packet
    test_proc_device_id_success = "device001" # This ID is in TRUSTED_DEVICE_SIGNATURES for receive_data_packet
    print(f"\nProcessing packet for {test_proc_device_id_success} (expect success):")
    try:
        simulated_packet_data = receive_data_packet(test_proc_device_id_success)
        
        # For this test, we need to generate a *correct* SHA256 signature for simulated_packet_data
        # because authenticate_signature uses SHA256.
        data_str_for_sig_gen = json.dumps(simulated_packet_data, sort_keys=True, separators=(',', ':'))
        hasher_for_test = hashlib.sha256()
        hasher_for_test.update(data_str_for_sig_gen.encode('utf-8'))
        correct_packet_signature = hasher_for_test.hexdigest()

        processed_data = process_iot_packet(test_proc_device_id_success, correct_packet_signature, simulated_packet_data)
        if processed_data:
            print(f"Successfully processed data for {test_proc_device_id_success}: {processed_data}")
        else:
            print(f"Failed to process data for {test_proc_device_id_success} (signature mismatch or other issue).")

    except ConnectionError as e:
        print(f"Error during setup for {test_proc_device_id_success} test: {e}")

    # Test Case 2: Failed processing (incorrect signature)
    test_proc_device_id_fail = "device001" # Using a device that receive_data_packet knows
    print(f"\nProcessing packet for {test_proc_device_id_fail} (expect failure - bad signature):")
    try:
        simulated_packet_data_for_fail = receive_data_packet(test_proc_device_id_fail)
        # Use an obviously incorrect signature (must be 64 hex chars for SHA256 comparison)
        incorrect_packet_signature = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff" 
        
        processed_data_fail = process_iot_packet(test_proc_device_id_fail, incorrect_packet_signature, simulated_packet_data_for_fail)
        if processed_data_fail:
            # This path should ideally not be taken if the signature is indeed incorrect.
            print(f"Successfully processed data for {test_proc_device_id_fail}: {processed_data_fail} (UNEXPECTED outcome)")
        else:
            print(f"Correctly failed to process data for {test_proc_device_id_fail} due to signature mismatch.")

    except ConnectionError as e:
        print(f"Error during setup for {test_proc_device_id_fail} test: {e}")

    # Test Case 3: Attempt to get data from an untrusted device via receive_data_packet
    test_proc_device_id_untrusted = "device003" # This ID is NOT in TRUSTED_DEVICE_SIGNATURES
    print(f"\nAttempting to receive packet for untrusted device {test_proc_device_id_untrusted} (expect ConnectionError):")
    try:
        # This call is expected to fail and raise ConnectionError
        simulated_data_untrusted = receive_data_packet(test_proc_device_id_untrusted)
        
        # The following lines would only execute if receive_data_packet unexpectedly succeeded.
        # For completeness, if it did, we'd need a signature to test process_iot_packet.
        # However, the primary test here is the ConnectionError.
        print(f"Received data for {test_proc_device_id_untrusted}: {simulated_data_untrusted} (UNEXPECTED - should have failed)")
        # dummy_sig_for_untrusted = hashlib.sha256(json.dumps(simulated_data_untrusted, sort_keys=True).encode('utf-8')).hexdigest()
        # process_iot_packet(test_proc_device_id_untrusted, dummy_sig_for_untrusted, simulated_data_untrusted)

    except ConnectionError as e:
        print(f"Correctly failed to receive packet for {test_proc_device_id_untrusted} as expected: {e}")
        # In a real scenario, process_iot_packet would not be called if packet reception fails.

    # Example usage for authenticate_signature:
    sample_data = "This is some important IoT data."
    # Pre-calculate a signature for the sample_data
    # In a real scenario, the signature would be generated by the device or a trusted source
    hasher_for_example = hashlib.sha256()
    hasher_for_example.update(sample_data.encode('utf-8'))
    correct_signature = hasher_for_example.hexdigest()
    incorrect_signature = "thisisnotthecorrectsignature"

    print(f"\nAttempting to authenticate data: '{sample_data}'")
    print(f"With correct signature: {correct_signature}")
    if authenticate_signature(sample_data, correct_signature):
        print("Authentication successful: Signatures match.")
    else:
        print("Authentication failed: Signatures do not match.")

    print(f"\nAttempting to authenticate data: '{sample_data}'")
    print(f"With incorrect signature: {incorrect_signature}")
    if authenticate_signature(sample_data, incorrect_signature):
        print("Authentication successful: Signatures match.")
    else:
        print("Authentication failed: Signatures do not match.")

    print(f"\nAttempting to retrieve data for {test_device_id_invalid}:")
    try:
        data = retrieve_iot_data(test_device_id_invalid)
        print(f"Data for {test_device_id_invalid}: {data}")
    except ConnectionError as e:
        print(e)
