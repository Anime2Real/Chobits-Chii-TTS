"""本地浏览器文本听辨修正工具.

读取 data/trimmed/manual/text_review.csv 与 reviewed/wavs/, 逐条试听并填写:
  - keep: 当前文本正确;
  - fix:  填写听辨后的修正文本;
  - drop: 音频不适合训练, 从最终训练集排除.

若 manual/asr_candidates.csv 存在, 页面会同时显示 Qwen3-ASR 与 Whisper
候选、中文辅助翻译和一致度. 结果仍写回 text_review.csv; 服务只监听
127.0.0.1, 无鉴权.
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "trimmed"
UI_PATH = Path(__file__).with_name("review_text.html")
FIELDS = ["priority", "file", "current_text", "reason", "verified_text",
          "action", "reviewed_at"]
ACTIONS = {"keep", "fix", "drop"}


class TextReviewStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.review_path = self.data_dir / "manual" / "text_review.csv"
        self.audio_dir = self.data_dir / "reviewed" / "wavs"
        if not self.review_path.is_file():
            raise FileNotFoundError(
                f"缺少 {self.review_path}; 请先运行 tools/trim_dataset.py 并生成文本复核队列"
            )
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(
                f"缺少 {self.audio_dir}; 请先运行 tools/build_reviewed_dataset.py"
            )
        self.asr_path = self.data_dir / "manual" / "asr_candidates.csv"
        self.lock = Lock()
        self.records = self._load()
        self.by_file = {r["file"]: r for r in self.records}
        self.asr_by_file = self._load_asr()

    def _load(self):
        with open(self.review_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        records = []
        seen = set()
        for row in rows:
            item = {key: row.get(key, "") or "" for key in FIELDS}
            if not item["file"] or item["file"] in seen:
                raise ValueError(f"{self.review_path}: 文件名为空或重复")
            if item["action"] not in ("",) + tuple(ACTIONS):
                raise ValueError(f"{self.review_path}: {item['file']} action 非法")
            if not self._audio_path(item["file"]).is_file():
                raise ValueError(f"缺少音频: {self._audio_path(item['file'])}")
            seen.add(item["file"])
            records.append(item)
        return records

    def _load_asr(self):
        if not self.asr_path.is_file():
            return {}
        with open(self.asr_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return {row["file"]: row for row in rows if row.get("file")}

    def _audio_path(self, name):
        if name not in getattr(self, "by_file", {}) and getattr(self, "records", None):
            raise KeyError(f"未知音频: {name}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or ""):
            raise KeyError(f"非法音频名: {name!r}")
        # 已按文本排除的片段不会出现在 reviewed/wavs; 回退到人工或自动切边音频,
        # 以便复核页仍可撤销/查看已完成决定.
        candidates = [
            self.audio_dir / f"{name}.wav",
            self.data_dir / "manual" / "wavs" / f"{name}.wav",
            self.data_dir / "wavs" / f"{name}.wav",
        ]
        return next((path for path in candidates if path.is_file()), candidates[0])

    def state(self):
        with self.lock:
            rows = [dict(r) for r in self.records]
            asr = {name: dict(row) for name, row in self.asr_by_file.items()}
        for row in rows:
            row["asr"] = asr.get(row["file"])
        counts = {"total": len(rows), "keep": 0, "fix": 0, "drop": 0, "pending": 0}
        for row in rows:
            action = row["action"]
            if action in ("keep", "fix", "drop"):
                counts[action] += 1
            else:
                counts["pending"] += 1
        return {"records": rows, "stats": counts}

    def save(self, name, action, verified_text):
        if action not in ACTIONS:
            raise ValueError("action 必须是 keep/fix/drop")
        with self.lock:
            if name not in self.by_file:
                raise KeyError(f"未知音频: {name}")
            row = self.by_file[name]
            corrected = str(verified_text or "").strip()
            if action == "fix":
                if not corrected:
                    raise ValueError("修正文本不能为空")
                if corrected == row["current_text"]:
                    raise ValueError("修正文本与当前文本相同, 请使用保留原文")
                if any(ch in corrected for ch in "\r\n|"):
                    raise ValueError("修正文本不能包含换行或 '|'")
            else:
                corrected = ""
            row["action"] = action
            row["verified_text"] = corrected
            row["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            self._write()
            return dict(row)

    def reset(self, name):
        with self.lock:
            if name not in self.by_file:
                raise KeyError(f"未知音频: {name}")
            row = self.by_file[name]
            row["action"] = ""
            row["verified_text"] = ""
            row["reviewed_at"] = ""
            self._write()
            return dict(row)

    def _write(self):
        temp = self.review_path.with_suffix(".csv.tmp")
        with open(temp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.records)
        os.replace(temp, self.review_path)


class TextReviewHandler(BaseHTTPRequestHandler):
    server_version = "ChiiTextReview/1.0"
    store = None
    ui_bytes = b""

    def log_message(self, fmt, *args):
        return

    def _send(self, status, body, content_type, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/":
                self._send(HTTPStatus.OK, self.ui_bytes, "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            elif path == "/api/state":
                self._json(HTTPStatus.OK, self.store.state())
            elif path.startswith("/api/audio/"):
                self._serve_audio(path[len("/api/audio/"):])
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (ValueError, OSError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _serve_audio(self, name):
        data = self.store._audio_path(name).read_bytes()
        range_header = self.headers.get("Range", "")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "bad range"})
                return
            start_s, end_s = match.groups()
            if not start_s:
                suffix = int(end_s or 0)
                start = max(0, len(data) - suffix)
                end = len(data) - 1
            else:
                start = int(start_s)
                end = min(int(end_s) if end_s else len(data) - 1, len(data) - 1)
            if start > end or start >= len(data):
                self._json(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, {"error": "bad range"})
                return
            self._send(HTTPStatus.PARTIAL_CONTENT, data[start:end + 1], "audio/wav",
                       {"Content-Range": f"bytes {start}-{end}/{len(data)}",
                        "Accept-Ranges": "bytes"})
            return
        self._send(HTTPStatus.OK, data, "audio/wav", {"Accept-Ranges": "bytes"})

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1024 * 1024:
            raise ValueError("请求体为空或过大")
        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError:
            raise ValueError("请求体不是合法 JSON") from None
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            name = str(payload.get("file", ""))
            if path == "/api/save":
                row = self.store.save(name, str(payload.get("action", "")),
                                      payload.get("verified_text", ""))
                self._json(HTTPStatus.OK, {"record": row})
            elif path == "/api/reset":
                row = self.store.reset(name)
                self._json(HTTPStatus.OK, {"record": row})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (ValueError, OSError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def build_parser():
    parser = argparse.ArgumentParser(description="小叽文本听辨修正工具 (仅本机)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="trim_dataset.py 输出目录")
    parser.add_argument("--port", type=int, default=9878, help="监听端口 (默认 9878)")
    parser.add_argument("--open", action="store_true", help="启动后尝试打开浏览器")
    return parser


def main():
    args = build_parser().parse_args()
    if not UI_PATH.is_file():
        raise SystemExit(f"缺少界面文件: {UI_PATH}")
    store = TextReviewStore(Path(args.data_dir))
    TextReviewHandler.store = store
    TextReviewHandler.ui_bytes = UI_PATH.read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), TextReviewHandler)
    url = f"http://127.0.0.1:{args.port}/"
    stats = store.state()["stats"]
    print(f"文本复核: {url}")
    print(f"共 {stats['total']} 条, 待处理 {stats['pending']} 条, "
          f"已保留 {stats['keep']}, 已修正 {stats['fix']}, 已排除 {stats['drop']}")
    if store.asr_by_file:
        print(f"ASR 候选: {len(store.asr_by_file)} 条 (Qwen + Whisper + 中文翻译)")
    else:
        print("ASR 候选: 未找到 manual/asr_candidates.csv")
    print("Ctrl-C 停止")
    if args.open:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
