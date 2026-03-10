import random
import string


def view_own_profile(user_id: int):
    """
    Allows a user to view their own profile information.

    Looks up the profile in the global USER_PROFILES mapping by user_id.
    Returns the profile details if found; otherwise returns 'User not found'.
    """
    profiles = globals().get('USER_PROFILES')
    if isinstance(profiles, dict) and isinstance(user_id, int):
        return profiles.get(user_id, 'User not found')
    return 'User not found'


def generate_random_password(length: int = 12) -> str:
    """
    Generate a random password containing uppercase, lowercase, digits, and special characters.

    Args:
        length: Total length of the password. Must be >= 4.

    Returns:
        A randomly generated password string.

    Raises:
        TypeError: If length is not an integer.
        ValueError: If length is less than 4.
    """
    if not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length < 4:
        raise ValueError("length must be at least 4 to include all required character classes")

    uppercase = string.ascii_uppercase
    lowercase = string.ascii_lowercase
    digits = string.digits
    specials = string.punctuation

    rng = random.SystemRandom()

    # Ensure at least one character from each required class
    password_chars = [
        rng.choice(uppercase),
        rng.choice(lowercase),
        rng.choice(digits),
        rng.choice(specials),
    ]

    # Fill the remaining length with a mix of all character classes
    all_chars = uppercase + lowercase + digits + specials
    password_chars.extend(rng.choice(all_chars) for _ in range(length - 4))

    # Shuffle to avoid predictable placement of required characters
    rng.shuffle(password_chars)

    return ''.join(password_chars)
