def process_iot_data(device_id, data, approved_devices):
    if device_id in approved_devices:
        return True
    else:
        return False
