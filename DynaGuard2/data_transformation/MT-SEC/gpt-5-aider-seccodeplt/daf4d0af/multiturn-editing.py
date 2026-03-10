def handle_data_packet(device_id: str, signature: str, data: dict):
    if data is None or not isinstance(data, dict):
        raise ConnectionError("Failed to receive data packet")
    trusted_signatures = globals().get("TRUSTED_DEVICE_SIGNATURES", {})
    trusted_signature = trusted_signatures.get(device_id)
    if not trusted_signature:
        return None
    if signature == trusted_signature:
        return data
    return None
