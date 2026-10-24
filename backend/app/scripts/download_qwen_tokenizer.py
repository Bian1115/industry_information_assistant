"""从 ModelScope 下载并校验 Qwen3 tokenizer 资产。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests


BASE_URL = "https://modelscope.cn/models/Qwen/Qwen3-8B/resolve/master"
FILES = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}


def download_file(target_dir: Path, filename: str, expected_sha256: str) -> None:
    target = target_dir / filename
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
        return

    temporary = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    with requests.get(f"{BASE_URL}/{filename}", stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                digest.update(chunk)
                handle.write(chunk)

    actual = digest.hexdigest()
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{filename} SHA256 不匹配：期望 {expected_sha256}，实际 {actual}"
        )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, sha256 in FILES.items():
        download_file(args.output, filename, sha256)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
