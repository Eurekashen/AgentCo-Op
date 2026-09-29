"""Invoke upstream GeneAgent (main_cascade.GeneAgent) on one gene set, unmodified.

Usage: python run_geneagent.py <genes-comma-separated> <out.json>

Runs in the current working directory, which the caller makes a fresh temp dir: GeneAgent
writes its results by appending to hard-coded relative paths, and this script reads them back.
"""

import json
import sys
from pathlib import Path

GENEAGENT_DIR = Path(__file__).resolve().parent / "GeneAgent"


def main() -> None:
    genes, out_path = sys.argv[1], Path(sys.argv[2])
    for d in ("Outputs/GPT-4", "Outputs/GeneAgent/Cascade", "Verification Reports/Cascade"):
        Path(d).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(GENEAGENT_DIR))
    import main_cascade  # noqa: E402  (module import sets up AgentPhD and its tools)

    main_cascade.GeneAgent("query", genes)

    def read(path: str) -> str:
        p = Path(path)
        return p.read_text(errors="replace") if p.exists() else ""

    final = read("Outputs/GeneAgent/Cascade/MsigDB_Final_Response_GeneAgent.txt")
    first = read("Outputs/GPT-4/MsigDB_Response_GPT4.txt")
    verification = read("Verification Reports/Cascade/Claims_and_Verification_for_MsigDB.txt")
    error = "====There are an error" in final
    text = final.replace("//\n", "").strip()
    process = ""
    for line in text.splitlines():
        if line.strip().lower().startswith("process:"):
            process = line.split(":", 1)[1].strip()
            break
    out_path.write_text(json.dumps({
        "ok": bool(text) and not error,
        "process": process,
        "final": text,
        "initial": first.replace("//\n", "").strip(),
        "verification": verification,
        # Set by the openai shim when a model call failed for infrastructure reasons.
        "infra_error": read("agentcoop_infra_error.txt").strip(),
    }))


if __name__ == "__main__":
    main()
