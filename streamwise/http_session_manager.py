"""
HTTP session manager for aiohttp ClientSession.
"""

import logging

from aiohttp import ClientSession
from aiohttp import ClientTimeout
from aiohttp import TCPConnector


client_session: ClientSession | None = None

# URL scheme used when constructing outbound service URLs (http or https).
# Call set_service_scheme("https") at startup to enable HTTPS for all
# service-to-service communication (health checks, file fetches, job
# submissions, etc.).
SERVICE_SCHEME: str = "http"


def set_service_scheme(scheme: str) -> None:
    """Set the URL scheme used for outbound service requests."""
    global SERVICE_SCHEME
    if scheme not in ("http", "https"):
        raise ValueError(f"Invalid service scheme: {scheme!r}. Must be 'http' or 'https'.")
    SERVICE_SCHEME = scheme


def create_client_session_instance() -> ClientSession:
    connector = TCPConnector(
        limit=100,
        limit_per_host=10,
        use_dns_cache=True)
    timeout = ClientTimeout(
        total=0.5,
        connect=0.5)
    return ClientSession(
        connector=connector,
        timeout=timeout)


async def startup() -> None:
    """Initialize sessions before the server starts."""
    global client_session
    if client_session is None or client_session.closed:
        logging.info("Creating aiohttp client session...")
        client_session = create_client_session_instance()


async def shutdown() -> None:
    """Cleanup tasks after server stops."""
    global client_session
    if client_session and not client_session.closed:
        logging.info("Closing aiohttp client session...")
        await client_session.close()
        client_session = None


async def get_global_session() -> ClientSession:
    """Get or create a global aiohttp ClientSession for reuse."""
    global client_session
    if client_session is None or client_session.closed:
        client_session = create_client_session_instance()
    return client_session
