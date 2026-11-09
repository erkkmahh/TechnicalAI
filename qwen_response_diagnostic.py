"""One localhost request with a simple Turkish question; no PDF input."""

import json
from pathlib import Path
import re
import sys
import time
import urllib.request


PAYLOAD = {
    "model": "qwen3:4b",
    "prompt": "Python programlama dili nedir? Türkçe olarak iki cümlede açıkla.",
    "think": False,
    "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 600},
    "stream": False,
}
OUTPUT = Path(__file__).resolve().parent / "qwen_response_diagnostic.json"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = {"endpoint": "http://localhost:11434/api/generate",
              "request_payload": PAYLOAD, "model_requests": 0,
              "raw_response": None, "response": None, "thinking": None,
              "output_tokens": None, "completion_reason": None, "error": None}

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("Redirects forbidden")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(
        report["endpoint"], data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    try:
        report["model_requests"] = 1
        with opener.open(request, timeout=180) as response:
            report["raw_response"] = response.read().decode("utf-8")
        envelope = json.loads(report["raw_response"])
        text = envelope.get("response", "")
        report.update({"response": text, "thinking_field_present": "thinking" in envelope,
                       "thinking": envelope.get("thinking"),
                       "output_tokens": envelope.get("eval_count"),
                       "completion_reason": envelope.get("done_reason"),
                       "generation_complete": envelope.get("done") is True and envelope.get("done_reason") == "stop",
                       "assessment": {
                           "answer_is_turkish": None, "answer_is_complete": None,
                           "contains_process_narration": bool(re.search(
                               r"let['’]s|I need to|the user|first,|düşünmem|önce.*düşün", text, re.I)),
                           "status": "OFFLINE_REVIEW_REQUIRED",
                           "note": "Narration detection is a keyword hint; language and task completion require response review.",
                       }})
        print(text, flush=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps(report["error"], ensure_ascii=False), flush=True)
    report["execution_seconds"] = round(time.monotonic() - started, 3)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model_requests", "output_tokens", "completion_reason", "execution_seconds")}, ensure_ascii=False), flush=True)
    print(f"Yerel rapor: {OUTPUT.name}", flush=True)
    return 1 if report["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
