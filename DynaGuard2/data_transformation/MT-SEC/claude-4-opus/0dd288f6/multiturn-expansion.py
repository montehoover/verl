import hashlib
import json

CONFIG_DATABASE = {
    "database.conf": {
        "host": "localhost",
        "port": 5432,
        "username": "admin",
        "password": "secret123",
        "database_name": "myapp_db",
        "connection_timeout": 30,
        "max_connections": 100
    },
    "app.conf": {
        "app_name": "MyApplication",
        "version": "2.1.0",
        "debug_mode": True,
        "log_level": "INFO",
        "max_upload_size": 10485760,
        "session_timeout": 3600,
        "allowed_hosts": ["localhost", "127.0.0.1", "example.com"]
    },
    "cache.conf": {
        "cache_type": "redis",
        "redis_host": "localhost",
        "redis_port": 6379,
        "cache_ttl": 300,
        "max_cache_size": 1000
    },
    "email.conf": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "use_tls": True,
        "sender_email": "noreply@example.com",
        "sender_name": "MyApp Notifications"
    }
}

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}


def read_config_file(filename):
    if filename not in CONFIG_DATABASE:
        raise IOError(f"Configuration file '{filename}' not found")
    
    return CONFIG_DATABASE[filename].copy()


def validate_signature(config_data, expected_signature):
    try:
        computed_hash = hashlib.sha256(config_data.encode('utf-8')).hexdigest()
        return computed_hash == expected_signature
    except:
        return False


def fetch_config_data(configuration_name: str):
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def apply_config_data(configuration_name: str, config_signature: str):
    config_dict = fetch_config_data(configuration_name)
    config_str = json.dumps(config_dict, sort_keys=True)
    
    if validate_signature(config_str, config_signature):
        return config_dict
    else:
        return None
