"""Exactly one native localhost Ollama chat request, with no PDF content."""

import json
from pathlib import Path
import re
import sys
import time
import urllib.request


ENDPOINT = "http://localhost:11434/api/chat"
PAYLOAD = {
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "Python programlama dili nedir? Türkçe olarak iki cümlede açıkla."}],
    "think": False,
    "stream": False,
    "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 600},
}
OUTPUT = Path(__file__).resolve().parent / "qwen_chat_diagnostic.json"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = {"endpoint": ENDPOINT, "request_payload": PAYLOAD, "model_requests": 0,
              "raw_response": None, "message_content": None, "message_thinking": None,
              "thinking_field_present": None, "completion_reason": None,
              "output_tokens": None, "error": None}

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("Redirects forbidden; localhost only")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(ENDPOINT, data=json.dumps(PAYLOAD).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    try:
        report["model_requests"] = 1
        with opener.open(request, timeout=180) as response:
            report["raw_response"] = response.read().decode("utf-8")
        envelope = json.loads(report["raw_response"])
        message = envelope["message"]
        content = message.get("content", "")
        report.update({"message_content": content, "message_thinking": message.get("thinking"),
                       "thinking_field_present": "thinking" in message,
                       "completion_reason": envelope.get("done_reason"),
                       "output_tokens": envelope.get("eval_count"),
                       "generation_complete": envelope.get("done") is True and envelope.get("done_reason") == "stop",
                       "assessment": {
                           "answer_is_turkish": None, "answer_is_complete": None,
                           "process_narration_in_content": bool(re.search(
                               r"let['’]s|let me|I need to|the user|first,|düşünmem", content, re.I)),
                           "thinking_markers_in_content": any(marker in content for marker in ("<think>", "</think>")),
                           "thinking_leakage_in_content": None,
                           "status": "OFFLINE_REVIEW_REQUIRED",
                           "note": "Keyword checks are hints; review content separately from any thinking field.",
                       }})
        print("message.content:", flush=True)
        print(content, flush=True)
        print("message.thinking:", flush=True)
        print(json.dumps(message.get("thinking"), ensure_ascii=False), flush=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps(report["error"], ensure_ascii=False), flush=True)
    report["execution_seconds"] = round(time.monotonic() - started, 3)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model_requests", "completion_reason", "output_tokens", "execution_seconds")}, ensure_ascii=False), flush=True)
    print(f"Yerel rapor: {OUTPUT.name}", flush=True)
    return 1 if report["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
