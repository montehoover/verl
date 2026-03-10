def receive_data_packet(device_id, pkt_data):
    """
    Placeholder function that retrieves data from devices.
    This should be replaced with the actual implementation.
    """
    # Mock implementation - returns the packet data as received
    return pkt_data


def process_iot_packet(device_id, pkt_data):
    # Pre-defined dictionary of trusted devices
    trusted_devices = {
        "device_001": True,
        "device_002": True,
        "device_003": True
    }
    
    try:
        data = receive_data_packet(device_id, pkt_data)
        # Check if device is trusted
        is_trusted = device_id in trusted_devices
        return is_trusted
    except Exception as e:
        raise ConnectionError(f"Failed to receive packet from device {device_id}: {str(e)}")
