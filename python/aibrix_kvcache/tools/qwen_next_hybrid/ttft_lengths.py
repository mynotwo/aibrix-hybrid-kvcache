"""TTFT by prompt length: recompute, local GPU prefix hit, AIBrix restore.

Per length, PAIRS unique prompts of exactly that many tokens. Each is sent
cold (recomputed, and its record saved), then after SETTLE seconds both
as a local prefix hit and, after resetting vLLM's prefix cache, as an
external restore; the two alternate in order (ABBA) across pairs.
The first pair is also the acceptance probe: its requests are named
bisect-{cold,local_hit,external} and its tokens go to s2-bisect.json.

Usage: python3 ttft_lengths.py PORT RUN_DIR PAIRS LEN [LEN ...]
Writes RUN_DIR/ttft.jsonl, RUN_DIR/s2-bisect.json and, last on stdout,
one JSON line of medians.
"""

import json
import random
import statistics
import sys
import time
import urllib.request

MODEL = "qwen-next"
MAX_TOKENS = 32
# A restore usually comes long after its save; don't time it behind one.
SETTLE = 3.0


def post(base, path, body=None, stream=False):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=1800)
    if stream:
        return resp
    with resp:
        raw = resp.read()
    return json.loads(raw) if raw.strip() else {}


def ttft(base, ids, request_id):
    """Seconds to the first streamed token, and the generated tokens."""
    body = {
        "model": MODEL,
        "prompt": ids,
        "max_tokens": MAX_TOKENS,
        "min_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "return_token_ids": True,
        "request_id": request_id,
    }
    start, first, tokens = time.perf_counter(), None, []
    with post(base, "/v1/completions", body, stream=True) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            new = json.loads(line[5:])["choices"][0].get("token_ids") or []
            if new and first is None:
                first = time.perf_counter() - start
            tokens.extend(new)
    return {"ttft_s": first, "tokens": tokens}


def prompt_ids(base, rng, length, pair):
    words = [
        f"L{length}p{pair}e{j} holds {rng.randrange(1000)} parts"
        for j in range(length // 12 + 20)
    ]
    text = f"Record {length}/{pair}: " + "; ".join(words)
    ids = post(base, "/tokenize", {"model": MODEL, "prompt": text})["tokens"]
    assert len(ids) >= length, (len(ids), length)
    return ids[:length]


def main(port, run_dir, pairs, *lengths):
    base, rng, rows = f"http://127.0.0.1:{port}", random.Random(20261018), []
    for length in map(int, lengths):
        for i in range(int(pairs)):
            ids = prompt_ids(base, rng, length, i)
            row = {"length": length, "pair": i}
            probe = not rows

            def name(arm):
                return f"bisect-{arm}" if probe else f"{arm}-{length}-{i}"

            row["cold"] = ttft(base, ids, name("cold"))
            time.sleep(SETTLE)
            order = ("local", "external") if i % 2 == 0 else (
                "external", "local")
            for arm in order:
                if arm == "external":
                    post(base, "/reset_prefix_cache")
                    time.sleep(0.2)
                row[arm] = ttft(base, ids, name(arm))
            rows.append(row)
            print(length, i, *(round(row[a]["ttft_s"] * 1000, 1)
                               for a in ("cold", "local", "external")),
                  flush=True)
    with open(f"{run_dir}/ttft.jsonl", "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    first = rows[0]
    with open(f"{run_dir}/s2-bisect.json", "w") as f:
        json.dump({
            "prompt_tokens": first["length"],
            "cold": first["cold"]["tokens"],
            "local_hit": first["local"]["tokens"],
            "external": first["external"]["tokens"],
        }, f)
    summary = {
        length: {
            f"{arm}_ttft_ms": round(1000 * statistics.median(
                r[arm]["ttft_s"] for r in rows if r["length"] == length), 1)
            for arm in ("cold", "local", "external")
        }
        for length in sorted({r["length"] for r in rows})
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:])
