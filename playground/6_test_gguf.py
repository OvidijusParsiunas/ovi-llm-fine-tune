"""Step 6 — TEST every shrunk size: how small can it go before it breaks?

    python 6_test_gguf.py        (run 5_shrink.sh first)

No PyTorch here: each GGUF file runs in llama.cpp's llama-server, exactly how it
would run on a Raspberry Pi or in Ollama. We ask each file the same questions
as before and score it, biggest file first.
"""

import json
import subprocess
import time
import urllib.request
from pathlib import Path

from common import is_correct
from data import FACTS, GENERAL

PORT = 8090
URL = f"http://127.0.0.1:{PORT}"
MELLOWFIN = [(f["test"], f["check"]) for f in FACTS]
GENERAL_QS = [(g["question"], g["check"]) for g in GENERAL]


def ask(question):
    """Send one chat message to llama-server (the same API shape as OpenAI's)."""
    body = json.dumps({"messages": [{"role": "user", "content": question}],
                       "temperature": 0, "max_tokens": 60}).encode()  # temperature 0 = greedy, like before
    request = urllib.request.Request(f"{URL}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"].strip()


def wait_until_ready():
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{URL}/health")  # errors until the model has loaded
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("llama-server didn't start")


files = sorted(Path("out/gguf").glob("*.gguf"), key=lambda p: -p.stat().st_size)
print(f"{'file':32} {'size':>7}   Mellowfin  General")
mistakes = []
for path in files:
    server = subprocess.Popen(["llama-server", "-m", str(path), "--port", str(PORT)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_until_ready()
        scores = []
        for questions in (MELLOWFIN, GENERAL_QS):
            right = 0
            for question, check in questions:
                reply = ask(question)
                if is_correct(reply, check):
                    right += 1
                else:
                    mistakes.append(f"  {path.name}: {question}\n      -> {reply}")
            scores.append(f"{right}/{len(questions)}")
    finally:
        server.terminate()
        server.wait()
    size = path.stat().st_size / 1e6
    print(f"{path.name:32} {size:5.0f} MB   {scores[0]:>9}  {scores[1]:>7}")

if mistakes:
    print("\nWrong answers:")
    print("\n".join(mistakes))
