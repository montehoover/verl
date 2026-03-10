from typing import Any


__all__ = ["get_document_content"]


def get_document_content(document_id: str) -> str:
    """
    Retrieve the content of a document by its ID from the global DOCUMENTS_DB.

    Args:
        document_id (str): The ID of the document.

    Returns:
        str: The content of the document.

    Raises:
        TypeError: If document_id is not a string.
        KeyError: If the document_id does not exist in DOCUMENTS_DB.
        NameError: If DOCUMENTS_DB is not defined in the environment.
    """
    if not isinstance(document_id, str):
        raise TypeError("document_id must be a string")

    # Access via globals() to avoid referencing an undefined name at import time
    try:
        db = globals()["DOCUMENTS_DB"]
    except KeyError:
        raise NameError("DOCUMENTS_DB is not defined") from None

    return db[document_id]
