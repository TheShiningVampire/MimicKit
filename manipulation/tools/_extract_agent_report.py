"""Extract the final assistant text from a Claude subagent jsonl transcript (strips the preamble line)."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
last = None
for ln in open(src):
    try: d = json.loads(ln)
    except Exception: continue
    m = d.get("message", {})
    if d.get("type") == "assistant" and m.get("role") == "assistant":
        for c in m.get("content", []):
            if c.get("type") == "text" and len(c["text"]) > 2000:
                last = c["text"]
assert last, "no long text found"
# drop the one-line preamble before the first '---' or '#'
i = last.find("\n#")
if 0 < i < 300: last = last[i+1:]
open(dst, "w").write(last.rstrip() + "\n")
print(dst, len(last.split()), "words")
