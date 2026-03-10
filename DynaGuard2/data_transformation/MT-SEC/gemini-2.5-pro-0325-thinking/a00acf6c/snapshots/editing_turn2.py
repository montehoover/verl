import random
import string

# Global data structure for user profiles
# Example: user_profiles = {1: {"name": "Alice", "email": "alice@example.com"}, 2: {"name": "Bob", "email": "bob@example.com"}}
user_profiles = {}

def view_own_profile(user_id: int):
    """
    Allows a user to view their own profile details.

    Args:
        user_id: The ID of the user.

    Returns:
        The profile details if the user exists, otherwise "Profile not found".
    """
    if user_id in user_profiles:
        return user_profiles[user_id]
    else:
        return "Profile not found"

def generate_random_password(length: int = 12):
    """
    Generates a random password string.

    Args:
        length: The desired length of the password. Defaults to 12.

    Returns:
        A randomly generated password string.
    """
    if length < 4:
        raise ValueError("Password length must be at least 4 to include all character types.")

    # Define character sets
    uppercase_letters = string.ascii_uppercase
    lowercase_letters = string.ascii_lowercase
    digits = string.digits
    special_characters = string.punctuation

    # Ensure the password contains at least one of each character type
    password = [
        random.choice(uppercase_letters),
        random.choice(lowercase_letters),
        random.choice(digits),
        random.choice(special_characters)
    ]

    # Fill the rest of the password length with a mix of all characters
    all_characters = uppercase_letters + lowercase_letters + digits + special_characters
    remaining_length = length - 4
    for _ in range(remaining_length):
        password.append(random.choice(all_characters))

    # Shuffle the password list to ensure randomness and join to form the string
    random.shuffle(password)
    return "".join(password)
