from typing import FrozenSet

# Predefined list of valid document links.
# Populate this set with the exact URLs that should be considered valid.
VALID_DOCUMENT_LINKS: FrozenSet[str] = frozenset({
    # "https://example.com/docs/guide.pdf",
    # "https://example.com/docs/manual.html",
})


def document_exists(doc_link: str) -> bool:
    """
    Return True only if doc_link is in the predefined set of valid document links.
    Otherwise return False.
    """
    return doc_link in VALID_DOCUMENT_LINKS
