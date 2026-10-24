"""独立进程复用生产Qwen引用核验器；跳过业务包初始化，不加载业务图或数据库。"""
import json
from pathlib import Path
import sys
import types

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND / "app"))
package = types.ModuleType("service")
package.__path__ = [str(BACKEND / "app/service")]
sys.modules["service"] = package

from service.context_policy import runtime_nli_settings
from service.nli_verifier import verify_claims_sync

if __name__ == "__main__":
    request, response = map(Path, sys.argv[1:3])
    data = json.loads(request.read_text())
    settings = runtime_nli_settings()
    result = verify_claims_sync(data["claims"], settings)
    result["settings"] = settings.model_dump()
    response.write_text(json.dumps(result, ensure_ascii=False, indent=2))
