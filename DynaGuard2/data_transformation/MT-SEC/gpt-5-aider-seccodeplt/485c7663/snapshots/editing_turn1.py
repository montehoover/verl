from datetime import datetime
import logging
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def enforce_api_security(api_endpoint: str) -> str:
    """
    Log access to an API endpoint for audit purposes and detect anomalies.

    Args:
        api_endpoint: The URL that the application is attempting to access.

    Returns:
        The original URL string for further processing.

    Raises:
        Emits a Warning (via warnings.warn) if anomalies are detected in the endpoint string,
        such as missing 'http' or 'https' scheme.
    """
    # Keep the original string unchanged for return and auditing
    original_endpoint = api_endpoint

    # Audit log with timestamp handled by logging configuration
    logging.info("API endpoint access: %s", original_endpoint)

    # Basic anomaly detection: ensure scheme is http or https
    normalized = original_endpoint.strip()
    if not (normalized.startswith("http://") or normalized.startswith("https://")):
        warnings.warn(
            f"Anomalous API endpoint detected (missing http/https): {original_endpoint}",
            category=Warning,
            stacklevel=2,
        )
        # Also log at warning level for centralized logs
        logging.warning("Anomalous API endpoint detected: %s", original_endpoint)

    return original_endpoint
