INVOICES = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"}
}

def fetch_user_invoice(user_id: int, is_admin: bool):
    """
    Fetches an invoice. If is_admin is True, fetches the invoice for any user_id.
    If is_admin is False, it attempts to fetch the invoice for the user_id provided,
    assuming it's the requester's own ID.

    Args:
        user_id (int): The ID of the user whose invoice is targeted if is_admin is True.
                       The ID of the requester (fetching their own invoice) if is_admin is False.
        is_admin (bool): True if the requester has admin privileges.

    Returns:
        dict or str: The invoice dictionary if found and accessible.
                     'Invoice not found' if an admin requests a non-existent invoice.
                     'Unauthorized' if a non-admin attempts to fetch their invoice and it's not found (hence not accessible).
    """
    if is_admin:
        # Admin can fetch any invoice by user_id
        return INVOICES.get(user_id, 'Invoice not found')
    else:
        # Non-admin: user_id is their own ID. They are requesting their own invoice.
        # An invoice is accessible if it exists for their user_id.
        invoice = INVOICES.get(user_id)
        if invoice:
            return invoice  # Invoice found and accessible
        else:
            # Their own invoice not found, hence not accessible as per requirements
            return 'Unauthorized'
