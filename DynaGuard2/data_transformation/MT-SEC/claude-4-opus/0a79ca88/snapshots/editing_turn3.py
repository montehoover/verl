import requests

TRUSTED_USER_PROFILES = {}

def get_user_details(u_id, sig_code):
    # Validate cryptographic signature
    if u_id not in TRUSTED_USER_PROFILES or TRUSTED_USER_PROFILES[u_id] != sig_code:
        return None
    
    # Signature is valid, proceed to fetch profile
    try:
        response = requests.get(f"https://api.userservice.com/profiles/{u_id}")
        
        if response.status_code == 200:
            profile_data = response.json()
            return profile_data
        else:
            raise ConnectionError(f"Failed to fetch profile: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        # Handle connection errors, timeouts, etc.
        raise ConnectionError(f"Connection failed: {str(e)}")
