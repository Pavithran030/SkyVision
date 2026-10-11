"""Launch script for the SkyVision FastAPI modern web application."""

import argparse
import sys
import uvicorn

from cloud_rainfall.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Run SkyVision Modern Web Server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" ⛅ SkyVision: Ground Sky Recognition & Rainfall Nowcasting")
    print("=" * 60)
    print(f" Local Web Dashboard: http://{args.host}:{args.port}")
    print(f" Interactive API Docs: http://{args.host}:{args.port}/docs")
    print(" Press Ctrl+C to terminate.")
    print("=" * 60 + "\n")

    uvicorn.run(
        "cloud_rainfall.ui.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
