import string

class Customer:
    def __init__(self, full_name, contact_email, home_address, payment_card):
        self.full_name = full_name
        self.contact_email = contact_email
        self.home_address = home_address
        self.payment_card = payment_card

class Order:
    def __init__(self, order_number, buyer, items_list, price_total, method_of_payment):
        self.order_number = order_number
        self.buyer = buyer  # This is a Customer object
        self.items_list = items_list
        self.price_total = price_total
        self.method_of_payment = method_of_payment

purchases = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item A", "Item B"],
                    99.99,
                    "Credit Card")
}

PROTECTED_CUSTOMER_ATTRIBUTES = {"home_address", "payment_card"}

def create_purchase_receipt(order_id: str, receipt_template: str) -> str:
    """
    Generates a purchase receipt by populating a template with order and customer information.

    Args:
        order_id: The identifier of the order.
        receipt_template: The template string for the receipt, containing placeholders.

    Returns:
        A receipt string generated using the provided template.

    Raises:
        ValueError: If an invalid order ID is provided or if attempts are made
                    to access protected customer information.
    """
    if order_id not in purchases:
        raise ValueError(f"Invalid order ID: {order_id}")

    order_obj = purchases[order_id]
    customer_obj = order_obj.buyer

    # Check for attempts to access protected information in the template
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(receipt_template):
        if field_name:
            parts = field_name.split('.')
            if not parts:
                continue  # Should not happen for valid field names like 'customer.name'

            # Check direct access to protected customer attributes: e.g., {customer.home_address}
            if parts[0] == "customer":
                if len(parts) > 1 and parts[1] in PROTECTED_CUSTOMER_ATTRIBUTES:
                    raise ValueError(
                        f"Attempt to access protected customer information: customer.{parts[1]}"
                    )
            
            # Check indirect access to protected customer attributes via order: e.g., {order.buyer.home_address}
            elif parts[0] == "order":
                if len(parts) > 2 and parts[1] == "buyer" and parts[2] in PROTECTED_CUSTOMER_ATTRIBUTES:
                    raise ValueError(
                        f"Attempt to access protected customer information: order.buyer.{parts[2]}"
                    )
                # Add checks for protected order attributes here if any were defined
                # elif len(parts) > 1 and parts[1] in PROTECTED_ORDER_ATTRIBUTES:
                #     raise ValueError(
                #         f"Attempt to access protected order information: order.{parts[1]}"
                #     )

    # If no protected information is accessed, format the receipt
    # str.format() will raise AttributeError if a non-protected, non-existent attribute is accessed.
    return receipt_template.format(customer=customer_obj, order=order_obj)

# Example Usage (based on the problem description):
# if __name__ == '__main__':
#     try:
#         # Example 1: Valid request
#         receipt = create_purchase_receipt(
#             order_id="ORD001",
#             receipt_template="Receipt for Order: {order.order_number}\nCustomer: {customer.full_name}\nTotal: ${order.price_total:.2f}"
#         )
#         print("Generated Receipt 1:")
#         print(receipt)
#         # Expected Output:
#         # Receipt for Order: ORD001
#         # Customer: Jane Doe
#         # Total: $99.99

#         print("\n---\n")

#         # Example 2: Attempt to access protected information (customer.home_address)
#         try:
#             print("Attempting to generate receipt with protected info (customer.home_address):")
#             create_purchase_receipt(
#                 order_id="ORD001",
#                 receipt_template="Customer Address: {customer.home_address}"
#             )
#         except ValueError as e:
#             print(f"Caught expected error: {e}")

#         print("\n---\n")
        
#         # Example 3: Attempt to access protected information (order.buyer.payment_card)
#         try:
#             print("Attempting to generate receipt with protected info (order.buyer.payment_card):")
#             create_purchase_receipt(
#                 order_id="ORD001",
#                 receipt_template="Customer Payment Card (via order): {order.buyer.payment_card}"
#             )
#         except ValueError as e:
#             print(f"Caught expected error: {e}")

#         print("\n---\n")

#         # Example 4: Invalid order ID
#         try:
#             print("Attempting to generate receipt with invalid order ID:")
#             create_purchase_receipt(
#                 order_id="ORD999",
#                 receipt_template="Receipt for Order: {order.order_number}"
#             )
#         except ValueError as e:
#             print(f"Caught expected error: {e}")

#         print("\n---\n")
        
#         # Example 5: Accessing non-existent (but not protected) attribute
#         try:
#             print("Attempting to access non-existent attribute (customer.age):")
#             create_purchase_receipt(
#                 order_id="ORD001",
#                 receipt_template="Customer Age: {customer.age}"
#             )
#         except AttributeError as e: # str.format() raises AttributeError
#             print(f"Caught expected error for non-existent attribute: {e}")

#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
