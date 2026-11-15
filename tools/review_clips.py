"""本地浏览器音频复核工具.

读取 tools/trim_dataset.py 的派生目录, 展示波形与句中长停顿. 支持:
  - 试听自动切边后的音频;
  - 调整首尾并预览剪辑;
  - 勾选并压短长停顿 (两侧各保留 PAD 秒);
  - 保存、排除或撤销单条结果.

服务只监听 127.0.0.1, 无鉴权, 供本机人工复核使用.
"""

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import wave
from array import array
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import unquote, urlparse

import trim_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "trimmed"
UI_PATH = Path(__file__).with_name("review_clips.html")
INTERNAL_PAD_S = 0.075
MIN_OUTPUT_S = 0.10
MAX_PREVIEW_INTERVALS = 100

DECISION_FIELDS = [
    "file", "action", "start", "end", "removed_intervals",
    "output_s", "reviewed_at",
]


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是数字") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数字")
    return result


class ReviewStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.wav_dir = self.data_dir / "wavs"
        self.manual_dir = self.data_dir / "manual"
        self.manual_wav_dir = self.manual_dir / "wavs"
        self.decisions_path = self.manual_dir / "decisions.csv"
        self.report_path = self.data_dir / "trim_report.csv"
        self.queue_path = self.data_dir / "review_queue.csv"
        for required in (self.report_path, self.wav_dir):
            if not required.exists():
                raise FileNotFoundError(
                    f"缺少 {required}, 请先运行 python tools/trim_dataset.py"
                )
        self.manual_wav_dir.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.records = self._load_records()
        self.decisions = self._load_decisions()

    @staticmethod
    def _read_csv(path):
        if not path.exists():
            return []
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _parse_intervals(raw):
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except ValueError:
            return []
        intervals = []
        for item in parsed if isinstance(parsed, list) else []:
            try:
                start, end = float(item["start"]), float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(start) and math.isfinite(end) and end > start:
                intervals.append({"start": start, "end": end, "duration": end - start})
        return intervals

    def _load_records(self):
        queue = {row["file"]: row for row in self._read_csv(self.queue_path)}
        records = {}
        for row in self._read_csv(self.report_path):
            name = row["file"]
            queue_row = queue.get(name, {})
            records[name] = {
                "file": name,
                "text": row.get("text", ""),
                "status": row.get("status", ""),
                "duration": float(row.get("output_s", 0)),
                "source_duration": float(row.get("source_s", 0)),
                "head_removed": float(row.get("head_removed_s", 0)),
                "tail_removed": float(row.get("tail_removed_s", 0)),
                "internal": self._parse_intervals(row.get("internal_intervals", "")),
                "needs_review": name in queue,
                "priority": queue_row.get("priority", ""),
                "reasons": queue_row.get("reasons", "").split("+") if queue_row else [],
            }
        if not records:
            raise ValueError(f"{self.report_path} 中没有记录")
        return records

    def _load_decisions(self):
        decisions = {}
        for row in self._read_csv(self.decisions_path):
            name = row.get("file", "")
            if not name:
                continue
            decisions[name] = {
                "action": row.get("action", ""),
                "start": float(row.get("start", 0) or 0),
                "end": float(row.get("end", 0) or 0),
                "remove_intervals": self._parse_intervals(row.get("removed_intervals", "")),
                "output_s": float(row.get("output_s", 0) or 0),
                "reviewed_at": row.get("reviewed_at", ""),
            }
        return decisions

    def _record(self, name):
        record = self.records.get(name)
        if record is None:
            raise KeyError(f"未知音频: {name}")
        return record

    def _auto_path(self, name):
        self._record(name)
        return self.wav_dir / f"{name}.wav"

    def _manual_path(self, name):
        self._record(name)
        return self.manual_wav_dir / f"{name}.wav"

    def state(self):
        with self.lock:
            decisions = dict(self.decisions)
        items = []
        for name in sorted(self.records):
            item = dict(self.records[name])
            item["internal"] = [dict(run) for run in item["internal"]]
            item["decision"] = decisions.get(name)
            items.append(item)
        pending = sum(1 for item in items if item["needs_review"] and not item["decision"])
        reviewed = sum(1 for item in items if item["decision"])
        dropped = sum(1 for item in items if item["decision"] and item["decision"]["action"] == "drop")
        return {
            "records": items,
            "stats": {
                "total": len(items),
                "needs_review": sum(1 for item in items if item["needs_review"]),
                "pending": pending,
                "reviewed": reviewed,
                "dropped": dropped,
            },
        }

    def waveform(self, name, bins=1200):
        sample_rate, samples = trim_dataset.read_wav(self._auto_path(name))
        bins = max(1, min(int(bins), 4000))
        step = max(1, math.ceil(len(samples) / bins))
        peaks = []
        for offset in range(0, len(samples), step):
            frame = samples[offset:offset + step]
            if not frame:
                continue
            peaks.append([
                round(min(frame) / 32768, 4),
                round(max(frame) / 32768, 4),
            ])
        return {"file": name, "duration": len(samples) / sample_rate, "peaks": peaks}

    def _validated_intervals(self, name, values):
        record = self._record(name)
        if values is None:
            return []
        if not isinstance(values, list) or len(values) > MAX_PREVIEW_INTERVALS:
            raise ValueError("remove_intervals 必须是列表且不超过 100 项")
        result = []
        for item in values:
            if not isinstance(item, dict):
                raise ValueError("remove_intervals 的每一项必须是对象")
            start = _finite_float(item.get("start"), "停顿起点")
            end = _finite_float(item.get("end"), "停顿终点")
            if start < 0 or end > record["duration"] + 0.001 or end <= start:
                raise ValueError("停顿区间超出音频范围")
            result.append({"start": start, "end": end})
        result.sort(key=lambda run: run["start"])
        for previous, current in zip(result, result[1:]):
            if current["start"] < previous["end"]:
                raise ValueError("停顿区间不能重叠")
        return result

    def edit_audio(self, name, start, end, remove_intervals):
        record = self._record(name)
        start = _finite_float(start, "起点")
        end = _finite_float(end, "终点")
        if start < 0 or end > record["duration"] + 0.001 or end - start < MIN_OUTPUT_S:
            raise ValueError(f"剪辑范围须落在 0~{record['duration']:.2f}s 且至少 {MIN_OUTPUT_S:.2f}s")
        intervals = self._validated_intervals(name, remove_intervals)
        sample_rate, samples = trim_dataset.read_wav(self._auto_path(name))
        first = round(start * sample_rate)
        last = min(len(samples), round(end * sample_rate))
        source = samples[first:last]

        pieces = []
        cursor = 0
        for run in intervals:
            cut_start = max(start, run["start"] + INTERNAL_PAD_S)
            cut_end = min(end, run["end"] - INTERNAL_PAD_S)
            if cut_end <= cut_start:
                continue
            rel_start = round((cut_start - start) * sample_rate)
            rel_end = round((cut_end - start) * sample_rate)
            if rel_start > cursor:
                pieces.append(source[cursor:rel_start])
            cursor = max(cursor, rel_end)
        if cursor < len(source):
            pieces.append(source[cursor:])

        output = array("h")
        for piece in pieces:
            output.extend(piece)
        if len(output) < round(MIN_OUTPUT_S * sample_rate):
            raise ValueError("删除停顿后音频过短")
        return sample_rate, output

    @staticmethod
    def wav_bytes(sample_rate, samples):
        buffer = io.BytesIO()
        payload = array("h", samples)
        if sys.byteorder != "little":
            payload.byteswap()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(payload.tobytes())
        return buffer.getvalue()

    def preview(self, name, start, end, remove_intervals):
        sample_rate, samples = self.edit_audio(name, start, end, remove_intervals)
        return self.wav_bytes(sample_rate, samples)

    def save(self, name, action, start, end, remove_intervals):
        record = self._record(name)
        if action not in {"keep", "edit", "drop"}:
            raise ValueError("action 必须是 keep/edit/drop")
        with self.lock:
            output_path = self._manual_path(name)
            if action == "drop":
                output_s = 0.0
                removed = []
                start = end = 0.0
                output_path.unlink(missing_ok=True)
            elif action == "keep":
                start, end = 0.0, record["duration"]
                removed = []
                sample_rate, samples = trim_dataset.read_wav(self._auto_path(name))
                trim_dataset.write_wav(output_path, samples, sample_rate)
                output_s = record["duration"]
            else:
                removed = self._validated_intervals(name, remove_intervals)
                sample_rate, samples = self.edit_audio(name, start, end, removed)
                trim_dataset.write_wav(output_path, samples, sample_rate)
                output_s = len(samples) / sample_rate
                start, end = float(start), float(end)

            decision = {
                "action": action,
                "start": round(start, 3),
                "end": round(end, 3),
                "remove_intervals": [
                    {"start": run["start"], "end": run["end"]} for run in removed
                ],
                "output_s": round(output_s, 3),
                "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            self.decisions[name] = decision
            self._write_decisions()
            return decision

    def reset(self, name):
        self._record(name)
        with self.lock:
            self.decisions.pop(name, None)
            self._manual_path(name).unlink(missing_ok=True)
            self._write_decisions()

    def _write_decisions(self):
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        temp = self.decisions_path.with_suffix(".csv.tmp")
        with open(temp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DECISION_FIELDS)
            writer.writeheader()
            for name in sorted(self.decisions):
                item = self.decisions[name]
                writer.writerow({
                    "file": name,
                    "action": item["action"],
                    "start": f"{item['start']:.3f}",
                    "end": f"{item['end']:.3f}",
                    "removed_intervals": json.dumps(
                        item["remove_intervals"], ensure_ascii=False, separators=(",", ":")
                    ),
                    "output_s": f"{item['output_s']:.3f}",
                    "reviewed_at": item["reviewed_at"],
                })
        os.replace(temp, self.decisions_path)


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "ChiiReview/1.0"
    store = None
    ui_bytes = b""

    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith("4"):
            print(f"[review] {self.address_string()} {fmt % args}", file=sys.stderr)

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
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

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
            elif path.startswith("/api/waveform/"):
                name = path[len("/api/waveform/"):]
                self._json(HTTPStatus.OK, self.store.waveform(name))
            elif path.startswith("/api/audio/"):
                name = path[len("/api/audio/"):]
                self._serve_audio(name)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (ValueError, OSError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _serve_audio(self, name):
        path = self.store._auto_path(name)
        data = path.read_bytes()
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
            chunk = data[start:end + 1]
            self._send(
                HTTPStatus.PARTIAL_CONTENT,
                chunk,
                "audio/wav",
                {"Content-Range": f"bytes {start}-{end}/{len(data)}", "Accept-Ranges": "bytes"},
            )
            return
        self._send(
            HTTPStatus.OK,
            data,
            "audio/wav",
            {"Accept-Ranges": "bytes"},
        )

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Content-Length 非法") from None
        if length <= 0 or length > 2 * 1024 * 1024:
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
            if path == "/api/preview":
                body = self.store.preview(
                    name, payload.get("start", 0), payload.get("end", 0),
                    payload.get("remove_intervals", []),
                )
                self._send(HTTPStatus.OK, body, "audio/wav")
            elif path == "/api/save":
                action = str(payload.get("action", "edit"))
                decision = self.store.save(
                    name, action, payload.get("start", 0), payload.get("end", 0),
                    payload.get("remove_intervals", []),
                )
                self._json(HTTPStatus.OK, {"decision": decision})
            elif path == "/api/reset":
                self.store.reset(name)
                self._json(HTTPStatus.OK, {"ok": True})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (ValueError, OSError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def build_parser():
    parser = argparse.ArgumentParser(description="小叽音频人工复核工具 (仅本机)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="trim_dataset.py 的输出目录")
    parser.add_argument("--port", type=int, default=9876, help="监听端口 (默认 9876)")
    parser.add_argument("--open", action="store_true", help="启动后尝试打开浏览器")
    return parser


def main():
    args = build_parser().parse_args()
    if not UI_PATH.is_file():
        raise SystemExit(f"缺少界面文件: {UI_PATH}")
    store = ReviewStore(Path(args.data_dir))
    ReviewHandler.store = store
    ReviewHandler.ui_bytes = UI_PATH.read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    url = f"http://127.0.0.1:{args.port}/"
    stats = store.state()["stats"]
    print(f"人工复核: {url}")
    print(
        f"待复核 {stats['pending']} / {stats['needs_review']} 条, "
        f"已处理 {stats['reviewed']} 条"
    )
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
