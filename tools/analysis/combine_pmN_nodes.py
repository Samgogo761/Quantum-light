#!/usr/bin/env python3
"""DEPRECATED entry point — do not use in production.

Production ±N pairing must go through:
  tools/analysis/combine_pmN_nodes_production.py

That path checks id/weight/I/phi/order, optional node SHA-256, and writes
weighted Je_bar / Jo_bar together with ICS/CS. The historical combiner only
matched id/order and emitted per-node Jo/Je without the weighted summary.
"""
from __future__ import annotations

import sys


def main() -> None:
    sys.stderr.write(
        "ERROR: combine_pmN_nodes.py is deprecated for production.\n"
        "Use: tools/analysis/combine_pmN_nodes_production.py\n"
        "(strict checks + weighted Je/Jo + ICS/CS).\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
