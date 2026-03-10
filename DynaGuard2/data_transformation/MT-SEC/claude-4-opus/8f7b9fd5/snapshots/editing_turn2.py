def process_iot_data(device_id, data_packet):
    known_devices = ['device001', 'device002', 'device003', 'device004', 'device005']
    
    if device_id not in known_devices:
        return None
    
    return data_packet
