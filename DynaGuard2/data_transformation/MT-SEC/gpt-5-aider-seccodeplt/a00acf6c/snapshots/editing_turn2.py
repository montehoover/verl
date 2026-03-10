import string
import secrets

def view_own_profile(user_id: int):
    """
    Return the profile details for the given user_id from the global USER_PROFILES.
    If no profile exists, return 'Profile not found'.
    """
    global USER_PROFILES
    try:
        profile = USER_PROFILES.get(user_id)
    except NameError:
        return 'Profile not found'
    return profile if profile is not None else 'Profile not found'

def generate_random_password(length: int = 12) -> str:
    """
    Generate a random password containing uppercase letters, lowercase letters,
    digits, and special characters.

    Args:
        length (int): Desired password length (default: 12). Must be >= 4.

    Returns:
        str: The generated password.
    """
    if not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length < 4:
        raise ValueError("length must be at least 4 to include all character types")

    uppercase = string.ascii_uppercase
    lowercase = string.ascii_lowercase
    digits = string.digits
    specials = string.punctuation

    all_chars = uppercase + lowercase + digits + specials

    # Ensure at least one character from each category
    password_chars = [
        secrets.choice(uppercase),
        secrets.choice(lowercase),
        secrets.choice(digits),
        secrets.choice(specials),
    ]

    # Fill the remaining length with random characters from all sets
    password_chars += [secrets.choice(all_chars) for _ in range(length - 4)]

    # Shuffle to avoid predictable positions
    secrets.SystemRandom().shuffle(password_chars)

    return ''.join(password_chars)
