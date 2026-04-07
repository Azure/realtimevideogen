#!/usr/bin/env python3

from run_httpserver import run_httpserver

import argparse
import logging
import asyncio

from wrapper_thinksound import ThinkSoundGeneration

import sys
sys.path.append("..")  # Add parent directory to path


async def main() -> None:
    parser = argparse.ArgumentParser(description="ThinkSound HTTP Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=18087, help="Port to bind to")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--use-half", action="store_true", help="Use half precision")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # Create ThinkSound generation instance
    generation = ThinkSoundGeneration(
        model_name="thinksound",
        use_half=args.use_half,
        sample_rate=args.sample_rate,
    )

    # Run HTTP server
    await run_httpserver(
        generation,
        host=args.host,
        port=args.port,
        return_response_headers={
            "Content-Type": "audio/wav"
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
