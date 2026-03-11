import aiofiles
import base64


def binary_to_base64(binary_data: bytes) -> str:
    """Converts binary data to a base64-encoded string."""
    if not isinstance(binary_data, bytes):
        raise TypeError(f"Expected bytes for binary_data, got {type(binary_data)}")
    base64_bytes = base64.b64encode(binary_data)
    base64_str = base64_bytes.decode('utf-8')
    return base64_str


def base64_to_binary(base64_str: str) -> bytes:
    """Converts a base64-encoded string to binary data."""
    if not isinstance(base64_str, str):
        raise TypeError(f"Expected str for base64_str, got {type(base64_str)}")
    base64_bytes = base64_str.encode('utf-8')
    binary_data = base64.b64decode(base64_bytes)
    return binary_data


async def save_base64_as_binary(
    file_path: str,
    base64_str: str
) -> str:
    assert isinstance(file_path, str)
    assert isinstance(base64_str, str)
    binary_data = base64_to_binary(base64_str)
    async with aiofiles.open(file_path, "wb") as file:
        await file.write(binary_data)
    return file_path


async def read_file_bytes(
    file_path: str
) -> bytes:
    """Read a file asynchronously and return its content as bytes."""
    if not isinstance(file_path, str):
        raise TypeError(f"Expected str for file_path, got {type(file_path)}")
    if not await aiofiles.os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    async with aiofiles.open(file_path, "rb") as file:
        content = await file.read()
    return content


async def read_file_base64(
    file_path: str
) -> str:
    """Read a file asynchronously and return its content as a base64-encoded string."""
    file_bytes = await read_file_bytes(file_path)
    return binary_to_base64(file_bytes)
