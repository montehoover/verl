def process_packet_data(id_device, packet_data):
    """
    Process data packets from IoT devices.
    
    Args:
        id_device (str): The device identifier
        packet_data (dict): The packet data dictionary
        
    Returns:
        bool: True if device is trusted, False otherwise
    """
    global trusted_devices
    
    try:
        # Check if device is in trusted list
        if id_device not in trusted_devices:
            return False
        
        # Device is trusted, return True
        return True
        
    except Exception as e:
        # Handle any connection or data access issues
        return False
