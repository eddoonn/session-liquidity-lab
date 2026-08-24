"""Push trade logs + summary to Discord via webhook (webhook URL in .env)."""
import os
import sys
import json
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path=os.path.join(HERE, ".env")):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def post_json(payload):
    req = urllib.request.Request(
        os.environ["DISCORD_WEBHOOK"],
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (lab-bot)"})
    with urllib.request.urlopen(req) as r:
        return r.status


def post_file(filepath, content_text):
    boundary = uuid.uuid4().hex
    fn = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-control; name="payload_json"\r\n'
        f"Content-Type: application/json\r\n\r\n"
        f"{json.dumps({'content': content_text})}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files[0]"; filename="{fn}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        os.environ["DISCORD_WEBHOOK"],
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "Mozilla/5.0 (lab-bot)"})
    with urllib.request.urlopen(req) as r:
        return r.status


SUMMARY = {
    "embeds": [{
        "title": "Session Liquidity Lab — 2026 OOS results",
        "color": 0x2f6fde,
        "fields": [
            {"name": "GOLD NY-Judas v3",
             "value": "103 trades · **+37.6R** · PF 2.18 · WR 69% · DD 3.1R · "
                      "5m-calibrated +37.5R", "inline": False},
            {"name": "USDJPY Asia-NY v3",
             "value": "101 trades · **+21.3R** · PF 1.84 · WR 77% · DD 4.9R · "
                      "5m-calibrated +21.3R", "inline": False},
            {"name": "Combined",
             "value": "**+58.9R** / 204 trades · maxDD 4.8R · execution-tax ≈ 0",
             "inline": False},
            {"name": "Protocol",
             "value": "train <2026-01-01 only · dual fill-model gates · "
                      "5-minute ground-truth calibration · costs included",
             "inline": False},
        ],
        "footer": {"text": "backtest conventions: naive hourly fills, costs included"},
    }]
}


def main():
    load_env()
    if "--summary-only" not in sys.argv:
        pass
    print("posting summary ...", post_json(SUMMARY))
    files = [
        ("results/detailed/trades2026_GOLD_NYJUDAS_v3.csv",
         "**GOLD NY-Judas v3** — all 2026 trades"),
        ("results/detailed/trades2026_USDJPY_BASE_v3.csv",
         "**USDJPY Asia-NY v3** — all 2026 trades"),
    ]
    for path, msg in files:
        full = os.path.join(HERE, path)
        print(f"posting {path} ...", post_file(full, msg))


if __name__ == "__main__":
    main()
