"""Record-and-replay for the web APIs external agents call (NCBI, Enrichr, g:Profiler, PubTator).

Loaded automatically because the shim directory is first on PYTHONPATH of the external-agent
subprocess. Replaying database responses makes a re-run of an external agent reproducible, which
controlled suffix replay needs; a fresh ``AGENTCOOP_HTTP_CACHE`` path gives a live run.
"""

import os

_path = os.environ.get("AGENTCOOP_HTTP_CACHE")
if _path:
    import requests_cache

    requests_cache.install_cache(
        _path,
        backend="sqlite",
        allowable_methods=("GET", "POST"),
        allowable_codes=(200,),
        expire_after=-1,
    )
