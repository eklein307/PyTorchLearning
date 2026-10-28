"""
Integrating the RUL model into another Python program WITHOUT importing it,
by talking to the CLI. The same pattern works from any language that can start a
process (C#, MATLAB, LabVIEW, Node, Rust, ...).

    python examples/call_from_python.py models/rul_gru.pt data/test.csv
"""
import json
import subprocess
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/rul_gru.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/test.csv"
CLI = [sys.executable, "-m", "pemfc_rul"]


def predict_once(csv_path):
    """One-shot: start a process, get JSON back. Simple; ~1-2 s per call (Python + torch start-up)."""
    res = subprocess.run(CLI + ["predict", "--model", MODEL, "--input", csv_path, "--mc-samples", "20"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip())
    return json.loads(res.stdout)


class RulServer:
    """Persistent: one process, many requests (milliseconds each). Use for monitoring loops."""

    def __init__(self, model):
        self.p = subprocess.Popen(CLI + ["serve", "--model", model], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = json.loads(self.p.stdout.readline())
        assert ready.get("ready"), ready

    def request(self, **req):
        self.p.stdin.write(json.dumps(req) + "\n")
        resp = json.loads(self.p.stdout.readline())
        if not resp["ok"]:
            raise RuntimeError(resp["error"])
        return resp

    def close(self):
        self.p.stdin.write('{"cmd": "quit"}\n')
        self.p.stdin.close()
        self.p.wait()


if __name__ == "__main__":
    for row in predict_once(DATA)[:3]:
        print("one-shot:", row)

    srv = RulServer(MODEL)
    print("model info:", srv.request(cmd="info")["info"]["model_name"])
    resp = srv.request(id="batch-1", csv=DATA, mc_samples=20)
    for row in resp["predictions"][:3]:
        print("served  :", row)
    # Rows can also be sent inline, e.g. straight from a data-acquisition buffer:
    #   srv.request(id="live", rows=[{"time_h": 0.0, "U1": 0.65, ...}, ...])
    srv.close()
