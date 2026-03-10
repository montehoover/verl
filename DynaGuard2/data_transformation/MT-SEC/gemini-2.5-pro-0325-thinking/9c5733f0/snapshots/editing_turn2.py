# Global dictionary of trusted devices.
# Populate this dictionary with device IDs that are considered trusted.
# Example: TRUSTED_DEVICES = {"device_id_1": "Device 1 Info", "device_id_2": "Device 2 Info"}
TRUSTED_DEVICES = {}

def process_packet_data(id_device: str, packet_data: dict) -> bool:
    """
    Verifies if a device is trusted and then attempts to process its packet data.

    Args:
        id_device: The identifier of the device.
        packet_data: A dictionary containing data for multiple devices,
                     where keys are device IDs and values are their
                     respective data packets.

    Returns:
        bool: True if the device is trusted and an attempt to process its
              data was made (regardless of the outcome of data processing).
              False if the device is not in the trusted list.
    """
    if id_device not in TRUSTED_DEVICES:
        # Device is not trusted.
        # Optionally, log this event:
        # print(f"Device '{id_device}' is not trusted. Access denied.")
        return False

    # Device is trusted, proceed to attempt processing its data.
    try:
        # Attempt to fetch the data for the given device ID from the packet_data collection
        device_specific_data = packet_data.get(id_device)

        if device_specific_data is not None:
            # Data for id_device exists in packet_data
            if isinstance(device_specific_data, dict):
                # Data is found and is a dictionary.
                # Actual processing of 'device_specific_data' would happen here
                # if the function were to do more than just return True/False.
                # For now, we just acknowledge the attempt.
                # Optionally, log successful fetch for trusted device:
                # print(f"Data for trusted device '{id_device}' found and is valid format.")
                pass
            else:
                # Data for id_device is found but is not a dictionary.
                # This is an unexpected format. Handle gracefully.
                # Optionally, log this situation:
                # print(f"Warning: Data for trusted device '{id_device}' is not a dictionary. Found type: {type(device_specific_data)}")
                pass
        else:
            # Data for id_device not found in packet_data.
            # This simulates a "failure in receiving the data packet" or "connection issue"
            # for a trusted device.
            # Optionally, log this situation:
            # print(f"Info: No data packet found for trusted device '{id_device}'.")
            pass
    except Exception as e:
        # Catch any other unexpected errors during the process for a trusted device.
        # Handle gracefully.
        # Optionally, log the error:
        # print(f"Error processing packet data for trusted device '{id_device}': {e}")
        pass

    # If the device was trusted, the function returns True,
    # indicating an attempt to process was made.
    return True
