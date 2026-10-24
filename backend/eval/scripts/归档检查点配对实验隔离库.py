"""归档本运行器自建隔离库；完成快照校验后可清理其用户、会话范围内的资源。"""
import argparse
import asyncio
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def archive(out, cleanup=False):
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    import psycopg
    from psycopg import sql

    manifest = load(out / "运行清单.json")
    schema = manifest["schema"]
    if not re.fullmatch(r"checkpoint_pair_[0-9a-f]{32}", schema):
        raise ValueError("不是本实验自建的schema")
    if manifest["status"] in {"运行中", "准备中"}:
        raise ValueError("实验仍在运行，拒绝归档或清理")
    archive_folder = out / "隔离资源归档"
    archive_folder.mkdir(exist_ok=True)
    record_file = archive_folder / "归档与清理记录.json"
    previous = load(record_file) if record_file.exists() else {}
    if previous.get("schema_dropped"):
        print(json.dumps({"directory": str(out), "status": "已归档并清理，无需重复"}, ensure_ascii=False))
        return
    # 不终止进程；只要本实验的已记录工作进程还在运行就拒绝操作。
    for path in out.glob("题目*/*/进程启动.json"):
        started = load(path)
        proc = subprocess.run(["ps", "-p", str(started["pid"]), "-o", "args="], capture_output=True, text=True)
        if str(out) in proc.stdout and "运行检查点恢复四组配对实验.py" in proc.stdout:
            raise ValueError("实验工作进程仍存活")

    params = dict(host=os.getenv("POSTGRES_HOST", "localhost"), port=os.getenv("POSTGRES_PORT", "5432"),
                  user=os.getenv("POSTGRES_USER", "postgres"), password=os.getenv("POSTGRES_PASSWORD", "postgres123"),
                  dbname=os.getenv("POSTGRES_DB", "industry_assistant"), connect_timeout=10)
    with psycopg.connect(**params, autocommit=True) as db:
        exists = db.execute("SELECT 1 FROM pg_namespace WHERE nspname=%s", (schema,)).fetchone()
        if not exists:
            raise ValueError("schema已不存在，不能补做缺失快照")
        users = db.execute(sql.SQL("SELECT id,username,email,is_active FROM {}.users").format(sql.Identifier(schema))).fetchall()
        if any(not name.startswith("eval_pair_") or not email.endswith("@eval.invalid") or active
               for _, name, email, active in users):
            raise ValueError("隔离库包含非本实验用户，拒绝清理")
        sessions = db.execute(sql.SQL("SELECT user_id,id FROM {}.chat_sessions").format(sql.Identifier(schema))).fetchall()
        if any(uid not in {r[0] for r in users} for uid, _ in sessions):
            raise ValueError("发现不属于实验用户的会话")

        # 仅补导出从未导出的阶段。故障前缀的已有快照绝不能在D续跑后重导出。
        missing = [p.parent for p in out.glob("题目*/*/运行身份.json") if not (p.parent / "工作历史.json").exists()]
        if missing:
            script = out / "冻结源码/backend/eval/scripts/运行检查点恢复四组配对实验.py"
            spec = importlib.util.spec_from_file_location("checkpoint_pair_archival_runner", script)
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            runner.configure(schema)
            for folder in missing:
                asyncio.run(runner.export_run(folder, schema))
                save(folder / "中断后补导出记录.json", {"exported_at": stamp(), "reason": "中断阶段缺失数据库导出，仅补齐原始记录，不修改执行状态"})

        tables = [r[0] for r in db.execute("SELECT tablename FROM pg_tables WHERE schemaname=%s ORDER BY tablename", (schema,)).fetchall()]
        counts = {t: db.execute(sql.SQL("SELECT count(*) FROM {}.{}").format(sql.Identifier(schema), sql.Identifier(t))).fetchone()[0] for t in tables}
        sources_path = archive_folder / "会话证据完整原文.jsonl.gz"
        with gzip.open(sources_path, "wt", encoding="utf-8") as handle:
            rows = db.execute(sql.SQL("SELECT * FROM {}.session_evidence_documents ORDER BY session_id,document_id").format(sql.Identifier(schema)))
            keys = [d.name for d in rows.description]
            for row in rows:
                handle.write(json.dumps(dict(zip(keys, row)), ensure_ascii=False, default=str) + "\n")

        dump_file = archive_folder / "隔离库原始快照.dump"
        dump_tmp = dump_file.with_suffix(".dump.tmp")
        with dump_tmp.open("wb") as handle:
            result = subprocess.run(["docker", "exec", "industry_postgres", "pg_dump", "--username", params["user"],
                "--dbname", params["dbname"], "--format", "custom", "--schema", schema, "--no-owner", "--no-privileges"],
                stdout=handle, stderr=subprocess.PIPE, timeout=180)
        if result.returncode:
            raise RuntimeError("pg_dump失败；隔离库保持原样")
        dump_tmp.replace(dump_file)
        with dump_file.open("rb") as handle:
            listing = subprocess.run(["docker", "exec", "-i", "industry_postgres", "pg_restore", "--list"],
                stdin=handle, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if listing.returncode:
            raise RuntimeError("pg_restore目录校验失败；隔离库保持原样")
        toc = listing.stdout.decode()
        if f"SCHEMA - {schema}" not in toc or any(f"TABLE DATA {schema} {table} " not in toc for table in tables):
            raise RuntimeError("快照目录缺表；隔离库保持原样")
        (archive_folder / "数据库快照目录.txt").write_text(toc)

        sys.path.insert(0, str(out / "冻结源码/backend/app"))
        from service.session_evidence_vectors import SessionEvidenceVectors
        from service.session_evidence_rag import SessionEvidenceSettings
        settings=SessionEvidenceSettings.from_env()
        vectors = SessionEvidenceVectors(settings.embedding_dimensions,settings.embedding_model)
        collection = vectors._collection(create=False)
        vector_file = archive_folder / "本次会话向量原始记录.jsonl.gz"
        vector_counts = {}
        with gzip.open(vector_file, "wt", encoding="utf-8") as handle:
            for uid, sid in sessions:
                count = 0
                if collection is not None:
                    expr=vectors._scope(uid,sid)
                    total=collection.query(expr=expr,output_fields=["count(*)"],consistency_level="Strong")[0]["count(*)"]
                    fields=["id", "user_id", "session_id", "topic_id", "document_id", "unit_type", "generation", "vector"]
                    if total<=16000:
                        seen=set()
                        for offset in range(0,total,500):
                            batch=collection.query(expr=expr,output_fields=fields,offset=offset,limit=min(500,total-offset),consistency_level="Strong")
                            for row in batch:
                                if row["id"] in seen:raise RuntimeError("向量分页重复，保留数据库")
                                seen.add(row["id"])
                                handle.write(json.dumps(row,ensure_ascii=False,default=lambda v:v.tolist() if hasattr(v,"tolist") else str(v))+"\n")
                        count=len(seen)
                        if count!=total:raise RuntimeError("向量归档数量不一致，保留数据库")
                        vector_counts[str(sid)]=count
                        continue
                    iterator = collection.query_iterator(expr=expr, batch_size=500,
                        output_fields=["id", "user_id", "session_id", "topic_id", "document_id", "unit_type", "generation", "vector"],
                        consistency_level="Strong", timeout=120)
                    try:
                        while True:
                            batch = iterator.next()
                            if not batch:
                                break
                            for row in batch:
                                handle.write(json.dumps(row, ensure_ascii=False,
                                    default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")
                                count += 1
                    finally:
                        iterator.close()
                vector_counts[str(sid)] = count

        record = {"archived_at": stamp(), "schema": schema, "database_table_rows": counts,
            "session_vector_rows": vector_counts,
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in [dump_file, sources_path, vector_file, archive_folder / "数据库快照目录.txt"]},
            "validation": "pg_restore --list可读取且覆盖全部表；未执行实际还原演练",
            "missing_phase_exports_completed": [str(p.relative_to(out)) for p in missing],
            "schema_dropped": False, "vectors_cleaned": False}
        save(record_file, record)
        if cleanup:
            for uid, sid in sessions:
                vectors.delete_session(uid, sid)
                if collection is not None:
                    remaining = collection.query(expr=vectors._scope(uid, sid), output_fields=["count(*)"], consistency_level="Strong")
                    if not remaining or remaining[0].get("count(*)") != 0:
                        raise RuntimeError("本实验会话向量清理校验失败，保留数据库")
            record.update(vectors_cleaned=True, vectors_cleaned_at=stamp())
            save(record_file, record)
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            record.update(schema_dropped=True, schema_dropped_at=stamp())
            save(record_file, record)
            manifest.update(schema_preserved_for_analysis=False, isolated_schema_archived_and_cleaned=True,
                            archive_record=str(record_file.relative_to(out)))
            save(out / "运行清单.json", manifest)
        print(json.dumps({"directory": str(out), "tables": len(tables), "vector_rows": sum(vector_counts.values()),
                          "schema_dropped": record["schema_dropped"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    archive(args.input.resolve(), args.cleanup)
