import requests

TRUSTED_USER_PROFILES = {}

def get_user_details(u_id, is_verified):
    # Check access control if verification is required
    if is_verified and u_id not in TRUSTED_USER_PROFILES:
        return None
    
    # If not verified or user is trusted, proceed to fetch profile
    try:
        response = requests.get(f"https://api.userservice.com/profiles/{u_id}")
        
        if response.status_code == 200:
            profile_data = response.json()
            return profile_data
        else:
            return None
    except requests.exceptions.RequestException:
        # Handle connection errors, timeouts, etc.
        return None
