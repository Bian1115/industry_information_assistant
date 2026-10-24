"""Docker 入门练习：只使用 Python 标准库的 HTTP 接口。"""

import json
import os
import platform
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            status = 200
            payload = {
                "message": os.getenv("DEMO_MESSAGE", "你好，我正在 Docker 容器里运行！"),
                "hostname": socket.gethostname(),
                "python": platform.python_version(),
            }
        elif path == "/health":
            status, payload = 200, {"status": "ok"}
        else:
            status, payload = 404, {"error": "接口不存在，请访问 / 或 /health"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("练习接口已启动，监听容器内的 8080 端口", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
