# Assume DOCUMENTS_DB is predefined, for example:
DOCUMENTS_DB = {
    "doc1": "This is the content of document 1.",
    "doc2": "Content for the second document.",
    "doc3": "Another document's interesting content."
}

def get_document_content(document_id: str) -> str:
    """
    Retrieves the content of a document given its ID.

    Args:
        document_id: The ID of the document (string).

    Returns:
        The content of the document as a string, or None if not found.
    """
    return DOCUMENTS_DB.get(document_id)
