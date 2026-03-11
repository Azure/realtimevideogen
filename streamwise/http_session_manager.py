"""
HTTP session manager for aiohttp ClientSession.
"""

import logging

from aiohttp import ClientSession
from aiohttp import ClientTimeout
from aiohttp import TCPConnector


client_session: ClientSession | None = None


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
