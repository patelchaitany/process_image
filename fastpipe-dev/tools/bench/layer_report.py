"""Summarise a trtexec --exportProfile JSON: top layers and any reformat/copy layers at the I/O boundary."""
import json, sys
for path in sys.argv[1:]:
    prof = json.load(open(path))
    rows = [r for r in prof if isinstance(r, dict) and "name" in r]
    total = sum(r.get("averageMs", 0) for r in rows)
    print(f"\n== {path}: {len(rows)} layers, {total:.3f} ms per run")
    for r in sorted(rows, key=lambda r: -r.get("averageMs", 0))[:12]:
        print(f"  {r['averageMs']:7.3f} ms {100 * r['averageMs'] / total:5.1f}%  {r['name'][:110]}")
    ref = [r for r in rows if any(k in r["name"].lower() for k in ("reformat", "copy", "cast"))]
    print(f"  reformat/copy/cast layers: {len(ref)}, {sum(r['averageMs'] for r in ref):.3f} ms")
    for r in ref[:8]:
        print(f"    {r['averageMs']:7.3f} ms  {r['name'][:110]}")
