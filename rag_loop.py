#!/usr/bin/env python3
"""相容入口：預設使用批次 JSONL 版 V2。"""

from rag_loop_v2 import main


if __name__ == "__main__":
    print("[相容模式] 建議直接使用 python rag_loop_v2.py 或 python rag_loop_v3.py")
    raise SystemExit(main())
