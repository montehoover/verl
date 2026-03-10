def verify_signature(document_id, signature, expected_signer):
    """
    Verify if a signature is valid for a given document.
    
    Args:
        document_id: The ID of the document to verify
        signature: The signature to verify
        expected_signer: The expected signer of the document
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    # Simulate signature verification logic
    # In a real implementation, this would use cryptographic verification
    if not signature or not expected_signer:
        return False
    
    # Simple mock verification - in reality this would be cryptographic
    expected_signature = f"{expected_signer}:{document_id}:signed"
    return signature == expected_signature


def check_document(document_id, signature, expected_signer):
    """
    Check if a document has a valid signature.
    
    Args:
        document_id: The ID of the document to check
        signature: The signature to verify
        expected_signer: The expected signer of the document
        
    Returns:
        dict: A dictionary with 'valid' (bool) and 'message' (str) keys
    """
    is_valid = verify_signature(document_id, signature, expected_signer)
    
    if is_valid:
        return {
            'valid': True,
            'message': f'Document {document_id} has a valid signature from {expected_signer}'
        }
    else:
        return {
            'valid': False,
            'message': f'Document {document_id} has an invalid signature'
        }


# Example usage
if __name__ == "__main__":
    # Test with valid signature
    result = check_document("DOC123", "alice:DOC123:signed", "alice")
    print(result)
    
    # Test with invalid signature
    result = check_document("DOC456", "wrong_signature", "bob")
    print(result)
