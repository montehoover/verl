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


def read_config_file(filename):
    if filename not in CONFIG_DATABASE:
        raise IOError(f"Configuration file '{filename}' not found")
    
    return CONFIG_DATABASE[filename].copy()
