from typing import Dict
import hashlib

# Example dictionary to simulate an external API response store
API_MESSAGES: Dict[str, str] = {
    "msg_001": "Welcome to the system!",
    "msg_002": "Your request has been processed.",
    "msg_003": "Please verify your email address.",
    "msg_100": "System maintenance scheduled at 02:00 UTC."
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


if __name__ == "__main__":
    # Example usage:
    print(fetch_api_message("msg_001"))  # Expected: "Welcome to the system!"

    # Check signatures
    message = fetch_api_message("msg_001")
    sha256_sig = hashlib.sha256(message.encode("utf-8")).hexdigest()
    print(check_signature(message, sha256_sig))            # Expected: True
    print(check_signature(message, f"sha256:{sha256_sig}"))  # Expected: True
    print(check_signature(message, "md5:deadbeef"))        # Expected: False

    # This will raise a KeyError to simulate a missing message ID
    # Uncomment to test:
    # print(fetch_api_message("msg_999"))
