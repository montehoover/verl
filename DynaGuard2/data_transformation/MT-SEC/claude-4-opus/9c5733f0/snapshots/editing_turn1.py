def process_packet_data(id_device, packet_data):
    """
    Process data packets from IoT devices.
    
    Args:
        id_device (str): The device identifier
        packet_data (dict): The packet data dictionary
        
    Returns:
        dict: The data fetched using the device ID
    """
    try:
        # Fetch the data using the device ID as key
        if id_device in packet_data:
            return packet_data[id_device]
        else:
            # Return empty dict if device ID not found
            return {}
    except Exception as e:
        # Handle any connection or data access issues
        return {}
