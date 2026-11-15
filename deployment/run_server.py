import os
import sys
import argparse
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDCN++ Deployment 2 Server (best_cdcnpp.pth)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port number (default: 8001)")
    args = parser.parse_args()

    print("=" * 70)
    print("  CDCN++ Face Anti-Spoofing Web Deployment 2")
    print("  Model: best_cdcnpp.pth (Validation Acc: 87.72%, Optimal Threshold: 0.56)")
    print("  Paper: Searching Central Difference Convolutional Networks (CVPR 2020)")
    print("=" * 70)
    print(f"  * Access the web UI at:  http://{args.host}:{args.port}")
    print(f"  * API Docs available at: http://{args.host}:{args.port}/docs")
    print("=" * 70)

    uvicorn.run("backend:app", host=args.host, port=args.port, reload=False, app_dir=CURRENT_DIR)
