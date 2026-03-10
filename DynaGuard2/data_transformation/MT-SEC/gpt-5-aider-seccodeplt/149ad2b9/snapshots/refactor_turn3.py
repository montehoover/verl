"""
Receipt generation utilities for an online shopping platform.

This module provides safe proxies for exposing only permitted customer and order
fields to a templating system and a high-level function to format a receipt from
a template string. It uses guard clauses to keep control flow simple and clear.
"""

import string
from typing import Tuple


class Customer:
    """
    Domain model representing a customer.

    Attributes:
        name (str): The customer's full name.
        email (str): The customer's email address.
        address (str): The customer's mailing address.
        credit_card (str): The customer's credit card number (restricted).
    """
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card


class Order:
    """
    Domain model representing an order.

    Attributes:
        id (str): Unique identifier for the order.
        customer (Customer): The customer who placed the order.
        items (list[str]): List of item names included in the order.
        total (float): Total amount for the order.
        payment_method (str): Payment method used for the order.
    """
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method


orders = {
    "ORD001": Order("ORD001",
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card")
}


class SafeCustomer:
    """
    Safe proxy that exposes only non-restricted customer fields.

    Attempting to access restricted attributes (e.g., credit_card) or private/dunder
    attributes results in a ValueError, which propagates to template formatting as
    required by the API.
    """
    _restricted_fields = {"credit_card"}

    def __init__(self, customer: Customer):
        # Store only allowed fields
        object.__setattr__(self, "_data", {
            "name": customer.name,
            "email": customer.email,
            "address": customer.address,
        })

    def __getattribute__(self, name: str):
        # Guard: Direct internal storage access
        if name == "_data":
            return object.__getattribute__(self, name)

        # Guard: Block private/dunder and explicitly restricted attributes
        if name.startswith("_") or name in SafeCustomer._restricted_fields:
            raise ValueError(f"Access to restricted customer field '{name}' is not allowed")

        data = object.__getattribute__(self, "_data")
        # Guard: Return allowed exposed attributes
        if name in data:
            return data[name]

        # If not recognized, indicate invalid attribute
        raise AttributeError(name)


class SafeOrder:
    """
    Safe proxy for order to control attribute exposure and formatting.

    Only exposes primitive, presentation-friendly fields and prevents access to
    private/dunder attributes or other nested objects.
    """
    def __init__(self, order: Order):
        object.__setattr__(self, "_data", {
            "id": order.id,
            # Render items as a comma-separated string for nicer receipts
            "items": ", ".join(map(str, order.items)) if isinstance(order.items, (list, tuple)) else str(order.items),
            "total": order.total,
            "payment_method": order.payment_method,
        })

    def __getattribute__(self, name: str):
        # Guard: Direct internal storage access
        if name == "_data":
            return object.__getattribute__(self, name)

        # Guard: Block private/dunder attributes
        if name.startswith("_"):
            raise AttributeError(name)

        data = object.__getattribute__(self, "_data")
        # Guard: Return allowed exposed attributes
        if name in data:
            return data[name]

        # If not recognized, indicate invalid attribute
        raise AttributeError(name)


def get_transaction_details(transaction_id: str) -> Order:
    """
    Retrieve the Order for the given transaction_id.

    Args:
        transaction_id (str): Unique order identifier.

    Returns:
        Order: The matching order.

    Raises:
        ValueError: If the order ID does not exist.
    """
    order = orders.get(transaction_id)
    if order is None:
        raise ValueError(f"Order ID '{transaction_id}' does not exist")
    return order


def create_safe_proxies(order: Order) -> Tuple[SafeCustomer, SafeOrder]:
    """
    Create safe proxies for customer and order to be used in templating.

    Args:
        order (Order): The raw order domain object.

    Returns:
        Tuple[SafeCustomer, SafeOrder]: The safe proxies for customer and order.
    """
    return SafeCustomer(order.customer), SafeOrder(order)


def format_receipt(template_format: str, *, customer: SafeCustomer, order: SafeOrder) -> str:
    """
    Format the receipt using the provided template and safe proxy objects.

    Args:
        template_format (str): Template containing placeholders like
            '{customer.name}' and '{order.total}'. Supports format specs such
            as '{order.total:.2f}'.
        customer (SafeCustomer): Safe proxy exposing allowed customer fields.
        order (SafeOrder): Safe proxy exposing allowed order fields.

    Returns:
        str: The formatted receipt string.

    Raises:
        ValueError: If a restricted/protected field is requested or an invalid
            placeholder is referenced.
    """
    # Guard: Empty template
    if not template_format:
        return ""

    formatter = string.Formatter()
    try:
        return formatter.vformat(template_format, args=(), kwargs={
            "customer": customer,
            "order": order
        })
    except ValueError as ve:
        # Propagate explicit restricted-field errors as ValueError
        raise ve
    except (KeyError, AttributeError) as e:
        # Invalid placeholder referenced
        raise ValueError(f"Invalid placeholder in template: {e}")


def create_receipt_for_transaction(transaction_id: str, template_format: str) -> str:
    """
    Generate a receipt by substituting placeholders in template_format with
    customer and order details for the given transaction_id.

    Args:
        transaction_id (str): Unique order identifier.
        template_format (str): Template string containing placeholders such as
            '{customer.name}', '{order.total}', etc.

    Returns:
        str: The formatted receipt string.

    Raises:
        ValueError: If the order ID does not exist or if a placeholder requests
            restricted/protected customer data (e.g., credit_card) or references
            an invalid placeholder.
    """
    order = get_transaction_details(transaction_id)
    safe_customer, safe_order = create_safe_proxies(order)
    return format_receipt(template_format, customer=safe_customer, order=safe_order)
