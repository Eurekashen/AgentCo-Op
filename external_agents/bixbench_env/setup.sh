#!/bin/bash
# One-time setup of the BixBench execution environment and the Finch harness.
#   bash external_agents/bixbench_env/setup.sh
# 1. conda env `bixbench`: native replica of futurehouse/bixbench:aviary-notebook-env (R 4.4 /
#    Bioconductor 3.20 instead of the image's 4.3 / 3.18: the pinned set does not solve on
#    osx-arm64; metaeuk, spades and busco/udocker are unavailable). Every BixBench arm runs its
#    code here. Exact solve: explicit-osx-arm64.txt.
# 2. Jupyter kernelspecs `python` and `ir` (the names Finch asks for) pointing into that env.
# 3. Finch (data-analysis-crow aea66fd) + BixBench (4931118) in a Python 3.12 venv.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXT="$(dirname "$HERE")"
conda env list | grep -q "^bixbench " || conda env create -y -f "$HERE/environment.yml"
ENV="$(conda env list | awk '$1=="bixbench"{print $NF}')"
chmod -R u+w "$ENV"
ln -sf iqtree3 "$ENV/bin/iqtree2"
# CellTypist for the COVID case's sites without published labels (v4 §5.1); --no-deps keeps the
# solved environment unchanged (its dependencies are already in it).
"$ENV/bin/python" -m pip install -q --no-deps celltypist==1.6.3
# jupyter_client resolves Finch's "python" to the native "python3" spec (its own interpreter),
# so the env's interpreter must be published under "python3" as well.
mkdir -p "$HERE/jupyter/kernels/python" "$HERE/jupyter/kernels/ir"
cat > "$HERE/jupyter/kernels/python/kernel.json" <<JSON
{"argv": ["$ENV/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
 "display_name": "Python 3 (ipykernel)", "language": "python",
 "env": {"PATH": "$ENV/bin:/usr/bin:/bin:/usr/sbin:/sbin"}}
JSON
rm -rf "$HERE/jupyter/kernels/python3" && cp -R "$HERE/jupyter/kernels/python" "$HERE/jupyter/kernels/python3"
cat > "$HERE/jupyter/kernels/ir/kernel.json" <<JSON
{"argv": ["$ENV/bin/R", "--slave", "-e", "IRkernel::main()", "--args", "{connection_file}"],
 "display_name": "R", "language": "R",
 "env": {"PATH": "$ENV/bin:/usr/bin:/bin:/usr/sbin:/sbin"}}
JSON
F="$EXT/finch"
[ -d "$F/data-analysis-crow" ] || git clone -q https://github.com/Future-House/data-analysis-crow.git "$F/data-analysis-crow"
[ -d "$F/BixBench" ] || git clone -q https://github.com/Future-House/BixBench.git "$F/BixBench"
(cd "$F/data-analysis-crow" && git checkout -q aea66fd)
(cd "$F/BixBench" && git checkout -q 4931118)
[ -x "$F/.venv/bin/python" ] || uv venv -q --python 3.12 "$F/.venv"
VIRTUAL_ENV="$F/.venv" uv pip install -q -e "$F/data-analysis-crow"
# Read-only, so agents cannot change the environment other runs execute in (installs land in
# their own workspace: agentcoop.agents.bix.exec_env).
chmod -R a-w "$ENV"
echo "bixbench env: $ENV"
