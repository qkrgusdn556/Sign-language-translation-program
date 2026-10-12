import argparse
import json
import math
import os
import shutil
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sign-samples.json"
LOCK = threading.Lock()


def valid_samples(samples):
    def vector(value):
        return isinstance(value, list) and len(value) in (126, 130, 136) and all(
            type(n) in (int, float) and math.isfinite(n) and abs(n) <= 100 for n in value
        )

    if not isinstance(samples, list) or len(samples) > 5000:
        return False
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("label"), str) or not 1 <= len(sample["label"].strip()) <= 40:
            return False
        sequence = sample.get("sequence")
        if not vector(sample.get("vector")) and not (
            isinstance(sequence, list) and 6 <= len(sequence) <= 32 and all(vector(frame) for frame in sequence)
        ):
            return False
    return True


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        resolved = Path(super().translate_path(path)).resolve()
        if resolved.is_relative_to(DATA.parent):
            return str(ROOT / ".private-data-unavailable")
        return str(resolved)

    def allowed(self):
        host = self.headers.get("Host", "")
        expected = f"127.0.0.1:{self.server.server_port}"
        localhost = f"localhost:{self.server.server_port}"
        origin = self.headers.get("Origin")
        return host in (expected, localhost) and (origin is None or origin == f"http://{host}")

    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed():
            return self.reply(403, {"error": "Local access only"})
        if self.path.split("?", 1)[0] == "/api/samples":
            try:
                with LOCK:
                    value = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else None
                if value is not None and not valid_samples(value.get("samples")):
                    raise ValueError("Invalid saved samples")
                return self.reply(200, {"samples": value["samples"] if value else None})
            except (OSError, ValueError, AttributeError) as error:
                return self.reply(500, {"error": str(error)})
        if self.path.split("?", 1)[0].startswith("/data"):
            return self.reply(403, {"error": "Use sample backup export"})
        return super().do_GET()

    def do_POST(self):
        if not self.allowed():
            return self.reply(403, {"error": "Local access only"})
        if self.path != "/api/samples":
            return self.reply(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64 * 1024 * 1024:
                return self.reply(413, {"error": "Invalid file size"})
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict) or not valid_samples(value.get("samples")):
                return self.reply(400, {"error": "Invalid samples; existing data is unchanged"})
            with LOCK:
                DATA.parent.mkdir(parents=True, exist_ok=True)
                temporary = DATA.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as output:
                    json.dump({"version": 3, "samples": value["samples"]}, output, ensure_ascii=False)
                    output.flush()
                    os.fsync(output.fileno())
                if DATA.exists():
                    shutil.copy2(DATA, DATA.with_suffix(".json.bak"))
                os.replace(temporary, DATA)
            return self.reply(200, {"saved": len(value["samples"])})
        except (OSError, ValueError) as error:
            return self.reply(500, {"error": str(error)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT)))
    print(f"http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
