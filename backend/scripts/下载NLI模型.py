"""下载固定版本的NLI模型；运行服务只加载本地文件，不在线下载。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
REVISION = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
WEIGHTS_SHA256 = "65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e"
FILES = ("config.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
         "spm.model", "tokenizer.json", "model.safetensors", "README.md")
DEFAULT_DIR = Path(__file__).resolve().parents[1] / ".cache" / "nli" / REVISION


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--direct", action="store_true", help="此下载进程不使用环境代理")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def download(name):
        path = args.output / name
        for attempt in range(3):
            try:
                session = requests.Session()
                session.trust_env = not args.direct
                with session.get(f"{args.endpoint.rstrip('/')}/{MODEL_ID}/resolve/{REVISION}/{name}",
                                 timeout=(10, 60), stream=True) as response:
                    response.raise_for_status()
                    temp = path.with_suffix(path.suffix + ".part")
                    with temp.open("wb") as f:
                        for chunk in response.iter_content(1024 * 1024):
                            f.write(chunk)
                if name == "model.safetensors" and digest(temp) != WEIGHTS_SHA256:
                    raise ValueError("权重SHA256与官方发布值不一致")
                temp.replace(path)
                result = {"file": name, "size": path.stat().st_size, "sha256": digest(path)}
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return result
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, FILES))
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "endpoint": args.endpoint,
                "official_weights_sha256": WEIGHTS_SHA256, "files": files}
    (args.output / "下载清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("模型已保存：" + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
