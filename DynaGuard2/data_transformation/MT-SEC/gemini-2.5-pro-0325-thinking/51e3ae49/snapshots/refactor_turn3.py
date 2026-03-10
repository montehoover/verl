import logging

# Configure basic logging to output to console
# This setup can be customized further (e.g., file output, different format)
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)

# Create a logger instance for this module
logger = logging.getLogger(__name__)

def _check_signature(document_content: str, signature: str) -> bool:
    """
    Checks if the provided signature is valid for the given document content.

    This is a pure function responsible solely for the signature verification logic.
    In a real-world scenario, this function would involve cryptographic operations
    to verify the signature against the document content using a specific key or algorithm.

    Args:
        document_content: The content of the document as a string.
        signature: The signature string to be validated.

    Returns:
        bool: True if the signature is deemed valid, False otherwise.
    """
    logger.debug(f"Initiating signature check. Content (first 30 chars): '{document_content[:30]}...', Signature: '{signature}'")
    # Placeholder for actual signature verification logic.
    # This would typically involve cryptographic libraries (e.g., PyNaCl, cryptography).
    # For demonstration, we'll use a simple string comparison.
    # IMPORTANT: Replace this with actual, secure signature validation logic.
    expected_signature = "signed_for:" + document_content  # Example: a very basic "signature"
    is_valid = (signature == expected_signature)

    if is_valid:
        logger.info("Signature verification successful.")
    else:
        logger.warning(f"Signature verification failed. Expected something like '{expected_signature}', got '{signature}'.")
    return is_valid

def validate_document(document: dict) -> bool:
    """
    Validates a given document based on its structure and signature.

    The function's signature remains (document: dict) -> bool as requested.
    It logs the validation attempt and its outcome.

    Args:
        document: A dictionary representing the document. It is expected
                  to contain at least 'id', 'content', and 'signature' keys.

    Returns:
        bool: True if the document passes all validation checks (structure, signature),
              False otherwise.
    """
    # Log the attempt to validate the document
    # Ensure document is a dictionary before trying to access 'id'
    doc_id = "UnknownID"
    if isinstance(document, dict) and 'id' in document:
        doc_id = document['id']
    elif isinstance(document, dict):
        doc_id = "ID_Not_Found"

    logger.info(f"Validation attempt for document ID: {doc_id}")

    # Basic structural validation: Check if document is a dictionary and has required keys
    if not isinstance(document, dict):
        logger.error(f"Validation failed for document ID: {doc_id}. Input is not a dictionary.")
        return False

    required_keys = ['id', 'content', 'signature']
    missing_keys = [key for key in required_keys if key not in document]
    if missing_keys:
        logger.error(f"Validation failed for document ID: {doc_id}. Missing required keys: {', '.join(missing_keys)}.")
        return False

    # Extract content and signature for validation
    document_content = document['content']
    signature = document['signature']

    # Perform signature validation using the extracted pure function
    logger.debug(f"Calling _check_signature for document ID: {doc_id}")
    is_signature_valid = _check_signature(document_content, signature)

    if not is_signature_valid:
        # _check_signature already logs details about signature failure
        logger.warning(f"Document ID: {doc_id} - Overall validation failed due to invalid signature.")
        return False

    # Placeholder for other potential validation steps
    # For example, schema validation, content policy checks, etc.
    # if not _check_content_policy(document_content):
    #     logger.warning(f"Document ID: {doc_id} - Content policy violation.")
    #     return False

    logger.info(f"Document ID: {doc_id} - All validation checks passed successfully.")
    return True

# Example usage to demonstrate functionality
if __name__ == "__main__":
    logger.info("Starting example document validations...")

    # Example 1: Valid document
    valid_document = {
        "id": "doc001",
        "content": "This is a valid document.",
        "signature": "signed_for:This is a valid document." # Matches our placeholder logic
    }
    logger.info(f"\n--- Validating Document: {valid_document.get('id')} ---")
    result_valid = validate_document(valid_document)
    print(f"Validation result for {valid_document.get('id')}: {result_valid}\n")

    # Example 2: Document with an invalid signature
    invalid_signature_document = {
        "id": "doc002",
        "content": "This document has an incorrect signature.",
        "signature": "wrong_signature"
    }
    logger.info(f"\n--- Validating Document: {invalid_signature_document.get('id')} ---")
    result_invalid_sig = validate_document(invalid_signature_document)
    print(f"Validation result for {invalid_signature_document.get('id')}: {result_invalid_sig}\n")

    # Example 3: Document missing a required key (e.g., 'content')
    missing_key_document = {
        "id": "doc003",
        "signature": "some_signature"
        # 'content' key is missing
    }
    logger.info(f"\n--- Validating Document: {missing_key_document.get('id')} ---")
    result_missing_key = validate_document(missing_key_document)
    print(f"Validation result for {missing_key_document.get('id')}: {result_missing_key}\n")

    # Example 4: Input is not a dictionary
    not_a_document = "This is just a string, not a document object."
    logger.info(f"\n--- Validating Non-Document Input ---")
    # We can't get an ID easily here, so validate_document will handle it
    result_not_dict = validate_document(not_a_document) # type: ignore
    print(f"Validation result for non-dictionary input: {result_not_dict}\n")

    logger.info("Example document validations finished.")
