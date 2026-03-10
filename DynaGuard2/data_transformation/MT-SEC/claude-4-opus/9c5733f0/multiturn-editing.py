def process_packet_data(id_device, packet_signature, packet_data):
    """
    Process data packets from IoT devices with cryptographic authentication.
    
    Args:
        id_device (str): The device identifier
        packet_signature (str): The cryptographic signature of the packet
        packet_data (dict): The packet data dictionary
        
    Returns:
        dict or None: The packet data if signature is valid, None otherwise
        
    Raises:
        ConnectionError: If there is an issue receiving data from the IoT device
    """
    global TRUSTED_DEVICE_SIGNATURES
    
    # Check if device exists in trusted signatures
    if id_device not in TRUSTED_DEVICE_SIGNATURES:
        return None
    
    # Validate the packet signature
    if packet_signature != TRUSTED_DEVICE_SIGNATURES[id_device]:
        return None
    
    # Signature is valid, return the packet_data that was passed in
    return packet_data
