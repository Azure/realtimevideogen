"""
Client timeout configurations for various services.
"""

from aiohttp import ClientTimeout


SERVICE_TIMEOUT = ClientTimeout(
    connect=2.0,
    sock_connect=2.0,
    sock_read=60.0,
)

SERVICE_MEDIUM_TIMEOUT = ClientTimeout(
    connect=5.0,
    sock_connect=5.0,
    sock_read=5 * 60.0,
)

SERVICE_LONG_TIMEOUT = ClientTimeout(
    connect=10.0,
    sock_connect=10.0,
    sock_read=10 * 60.0,
)

SERVICE_WARMUP_TIMEOUT = ClientTimeout(
    connect=10.0,
    sock_connect=10.0,
    sock_read=10 * 60.0,
)
