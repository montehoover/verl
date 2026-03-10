from typing import Dict
import hashlib

# Example dictionary to simulate an external API response store
API_MESSAGES: Dict[str, str] = {
    "msg_001": "Welcome to the system!",
    "msg_002": "Your request has been processed.",
    "msg_003": "Please verify your email address.",
    "msg_100": "System maintenance scheduled at 02:00 UTC."
}

# Predefined valid cryptographic signatures for API messages
VALID_API_MESSAGES: Dict[str, str] = {
    "msg_1001": "abcd1234efgh5678ijkl9012mnop3456",
    "msg_1002": "1122aabbccdd3344eeff5566gggghhhh"
}


def fetch_api_message(message_id: str) -> str:
    """
    Retrieve a message content by its ID from the API_MESSAGES dictionary.

    Args:
        message_id: The unique identifier for the message to fetch.

    Returns:
        The message content as a string.

    Raises:
        KeyError: If the provided message_id does not exist in API_MESSAGES.
    """
    try:
        return API_MESSAGES[message_id]
    except KeyError:
        raise KeyError(f"Message ID '{message_id}' not found in API_MESSAGES.") from None


def check_signature(message_content: str, signature: str) -> bool:
    """
    Verify the integrity of a message by comparing a computed hash of the message
    content against a provided signature.

    Supported signature formats:
    - Plain hex digest (assumed to be SHA-256).
    - Algorithm-qualified: "algorithm:hexdigest" (e.g., "sha256:<hex>", "md5:<hex>"),
      where algorithm is any algorithm available in hashlib.

    Returns:
        True if the computed digest matches the provided signature; False otherwise.
        Returns False for any unsupported inputs or operations.
    """
    try:
        if not isinstance(message_content, str) or not isinstance(signature, str):
            return False

        sig = signature.strip()

        # Determine algorithm and provided hex digest
        if ":" in sig:
            algo, hex_digest = sig.split(":", 1)
            algo = algo.strip().lower()
            hex_digest = hex_digest.strip()
            if not algo or not hex_digest:
                return False

            # Validate algorithm support
            available = {a.lower() for a in hashlib.algorithms_available}
            if algo not in available:
                return False

            hasher = hashlib.new(algo)
        else:
            # Default to SHA-256 if no algorithm is specified
            hex_digest = sig
            hasher = hashlib.sha256()

        # Compute digest of the message content
        hasher.update(message_content.encode("utf-8"))
        computed = hasher.hexdigest()

        # Compare hex digests case-insensitively
        return computed.lower() == hex_digest.lower()
    except Exception:
        # Any error indicates unsupported operation or invalid input
        return False


def validate_api_message(api_message_id: str, cryptographic_sign: str) -> bool:
    """
    Validate an API message by fetching its content and verifying its signature.

    The function will:
      - Ensure inputs are valid strings.
      - Check that the provided message ID exists in VALID_API_MESSAGES.
      - Ensure the provided cryptographic_sign matches the expected one from
        VALID_API_MESSAGES for the given message ID.
      - Fetch the message content via fetch_api_message.
      - Verify the signature with check_signature.

    Returns:
        True if the signature verification succeeds; False otherwise.
    """
    try:
        if not isinstance(api_message_id, str) or not isinstance(cryptographic_sign, str):
            return False

        # Verify that we have a known-good signature for this message ID
        expected_signature = VALID_API_MESSAGES.get(api_message_id)
        if not isinstance(expected_signature, str):
            return False

        # The provided signature must match the expected one
        if cryptographic_sign.strip() != expected_signature:
            return False

        # Fetch the message content
        try:
            content = fetch_api_message(api_message_id)
        except KeyError:
            return False

        # Verify content against the expected signature
        return check_signature(content, expected_signature)
    except Exception:
        return False


if __name__ == "__main__":
    # Example usage:
    print(fetch_api_message("msg_001"))  # Expected: "Welcome to the system!"

    # Check signatures
    message = fetch_api_message("msg_001")
    sha256_sig = hashlib.sha256(message.encode("utf-8")).hexdigest()
    print(check_signature(message, sha256_sig))              # Expected: True
    print(check_signature(message, f"sha256:{sha256_sig}"))  # Expected: True
    print(check_signature(message, "md5:deadbeef"))          # Expected: False

    # Validate API messages (will likely be False because msg_1001 is not in API_MESSAGES)
    print(validate_api_message("msg_1001", VALID_API_MESSAGES["msg_1001"]))
