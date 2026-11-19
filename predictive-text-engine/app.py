"""Try the model interactively.

    python app.py cli              # type a phrase, see suggestions
    python app.py web              # autocomplete box at http://127.0.0.1:8000
    python app.py generate         # print a few generated sentences

Options: --corpus PATH (default data/corpus.txt)  --n 3  --k 0.01
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ngram_lm import NgramLM

ROOT = Path(__file__).resolve().parent

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Predictive text</title>
<style>
 body{font:18px/1.5 system-ui,sans-serif;max-width:640px;margin:10vh auto;padding:0 16px;background:#fafafa;color:#222}
 textarea{width:100%;box-sizing:border-box;font:inherit;padding:12px;border:1px solid #bbb;border-radius:8px;min-height:7em}
 #sug{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
 button{font:inherit;padding:6px 14px;border:1px solid #888;border-radius:999px;background:#fff;cursor:pointer}
 button:hover{background:#eef}
 small{color:#666}
 @media(prefers-color-scheme:dark){body{background:#181818;color:#eee}textarea,button{background:#222;color:#eee;border-color:#555}}
</style>
<h1>Predictive text</h1>
<textarea id=box autofocus placeholder="start typing..."></textarea>
<div id=sug></div>
<p><small>A trigram language model trained from scratch on public-domain novels. Click a suggestion (or press Tab for the first).</small></p>
<script>
const box=document.getElementById('box'),sug=document.getElementById('sug');let t;
async function refresh(){
  const r=await fetch('/suggest?q='+encodeURIComponent(box.value)); const words=await r.json();
  sug.replaceChildren(...words.map(w=>{const b=document.createElement('button');b.textContent=w;b.onclick=()=>pick(w);return b}));
}
function pick(w){
  const m=box.value.match(/[A-Za-z']+$/);
  box.value=(m?box.value.slice(0,m.index):box.value)+w+' '; box.focus(); refresh();
}
box.addEventListener('input',()=>{clearTimeout(t);t=setTimeout(refresh,60)});
box.addEventListener('keydown',e=>{if(e.key==='Tab'&&sug.firstChild){e.preventDefault();sug.firstChild.click()}});
refresh();
</script>"""


def make_handler(lm):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/suggest":
                q = parse_qs(url.query).get("q", [""])[0]
                body, ctype = json.dumps(lm.complete(q, topk=5)).encode(), "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # keep the console quiet
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["cli", "web", "generate"])
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus.txt"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=float, default=0.01)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    if not Path(a.corpus).exists():
        sys.exit(f"{a.corpus} not found - run: python scripts/download_data.py")
    print(f"training n={a.n} Witten-Bell model on {a.corpus} ...", flush=True)
    lm = NgramLM(a.n, a.k, "witten-bell", min_count=2).fit(Path(a.corpus).read_text(encoding="utf-8"))

    if a.mode == "generate":
        for seed in range(5):
            print(f"{seed}: {lm.generate(seed=seed, max_len=30)}")
    elif a.mode == "cli":
        print("Type a phrase and press Enter (empty line to quit). Ending mid-word completes it.")
        while True:
            try:
                text = input("> ")
            except EOFError:
                break
            if not text:
                break
            print("   ", lm.complete(text, topk=5) or ["(no idea)"])
    else:
        server = HTTPServer(("127.0.0.1", a.port), make_handler(lm))
        print(f"open http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
