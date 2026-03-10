import logging

# Configure logging if not already configured by the application
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

logger = logging.getLogger(__name__)

USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}

INVOICES = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"}
}


def is_authorized(user_roles: dict, current_user_id: int, customer_id: int) -> bool:
    """
    Determine if the current user is authorized to access the requested customer's invoice.
    Admins can access any invoice; customers can access only their own.
    """
    actor = user_roles.get(current_user_id)
    if actor is None:
        return False
    return actor.get("role") == "admin" or current_user_id == customer_id


def get_invoice(invoices: dict, customer_id: int):
    """
    Retrieve the invoice for the specified customer_id.
    Returns the invoice dict if it exists, otherwise None.
    """
    return invoices.get(customer_id)


def download_invoice(current_user_id: int, customer_id: int):
    """
    Allows customers to download their own invoices and administrators to download any customer's invoice.

    Args:
        current_user_id (int): The user_id of the actor requesting to download the invoice.
        customer_id (int): The user_id of the customer whose invoice is requested.

    Returns:
        dict | str: The invoice data if authorized, else 'Unauthorized'.
    """
    authorized = is_authorized(USER_ROLES, current_user_id, customer_id)

    if not authorized:
        logger.info(
            "Invoice download request: actor_id=%s customer_id=%s authorized=%s",
            current_user_id, customer_id, False
        )
        return 'Unauthorized'

    invoice = get_invoice(INVOICES, customer_id)
    logger.info(
        "Invoice download request: actor_id=%s customer_id=%s authorized=%s invoice_found=%s",
        current_user_id, customer_id, True, invoice is not None
    )
    return invoice
