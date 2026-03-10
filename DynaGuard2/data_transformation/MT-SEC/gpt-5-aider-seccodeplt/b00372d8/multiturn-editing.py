def process_iot_packet(device_id: str, crypt_sig: str, pkt_data: dict) -> dict | None:
    """
    Receive a data packet from an IoT device after verifying a cryptographic signature.

    Args:
        device_id: The unique identifier of the device sending the packet.
        crypt_sig: The cryptographic signature provided with the packet.
        pkt_data: The packet data (not used for retrieval, included for compatibility).

    Returns:
        The received data packet as a dictionary if the signature is valid; otherwise, None.

    Raises:
        ConnectionError: If the packet cannot be received due to a connection issue.
    """
    # Validate signature against trusted signatures
    signatures = globals().get("TRUSTED_DEVICE_SIGNATURES", {})
    if not isinstance(signatures, dict):
        signatures = {}
    expected_sig = signatures.get(device_id)

    if not isinstance(crypt_sig, str) or expected_sig is None or crypt_sig != expected_sig:
        return None

    # Only attempt to receive data after signature validation
    try:
        receiver = globals().get("receive_data_packet")
        if not callable(receiver):
            raise ConnectionError("Data receiver unavailable")
        data = receiver(device_id)
    except ConnectionError:
        # Propagate connection-related errors as-is
        raise
    except (TimeoutError, OSError) as exc:
        # Normalize common connectivity failures to ConnectionError
        raise ConnectionError("Failed to receive packet due to a connection issue") from exc

    return data
