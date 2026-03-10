import hashlib # For placeholder signature verification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _verify_signature_pure(content_to_verify: str, signature: str, verification_key: str) -> bool:
    """
    Verifies the signature of the given content using the verification key.
    This is a pure function: its output depends only on its inputs, and it has no side effects.

    Args:
        content_to_verify: The content whose signature is to be verified.
        signature: The signature to verify.
        verification_key: The key to use for verification (e.g., a public key).
                         (Note: Not actively used in this simplified placeholder).

    Returns:
        True if the signature is valid, False otherwise.

    Note: This implementation uses a highly simplified placeholder for signature verification.
    In a real-world scenario, this would involve proper cryptographic algorithms.
    """
    if not content_to_verify or not signature: # Basic check for empty inputs
        return False
    
    # Placeholder: A real implementation would use cryptographic libraries
    # and the 'verification_key'.
    # For this example, we assume the signature is simply the SHA256 hash of the content.
    expected_hash = hashlib.sha256(content_to_verify.encode('utf-8')).hexdigest()
    
    return signature == expected_hash

def check_document(document_id: str, content: str, signature: str, public_key: str) -> bool:
    """
    Checks the validity of a document, including its signature.
    The main function's signature is kept unchanged as per requirements.
    Signature verification logic is delegated to a separate pure function.

    Args:
        document_id: The identifier of the document.
        content: The textual content of the document.
        signature: The cryptographic signature of the document.
        public_key: The public key to be used for verifying the signature.

    Returns:
        True if the document is valid (including signature), False otherwise.
    """
    # Step 1: Perform other document validation checks (e.g., format, schema, metadata)
    if not document_id: # Example: Document ID must be present
        return False
    if len(content) == 0: # Example: Content must not be empty
        return False
    
    # Step 2: Verify the signature using the extracted pure function
    # The 'public_key' is passed to the pure function as 'verification_key'.
    is_signature_correct = _verify_signature_pure(content, signature, public_key)

    if not is_signature_correct:
        logging.info(f"Document verification failed for document ID: {document_id}. Signature: {signature}. Reason: Invalid signature.")
        return False

    # If all checks pass (basic validation and signature)
    logging.info(f"Document verification successful for document ID: {document_id}. Signature: {signature}.")
    return True
