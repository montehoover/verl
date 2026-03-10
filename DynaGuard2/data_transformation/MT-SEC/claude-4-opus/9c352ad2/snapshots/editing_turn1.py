from urllib.parse import urlparse

def process_webhook(webhook_url: str) -> dict:
    parsed = urlparse(webhook_url)
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path
    }
