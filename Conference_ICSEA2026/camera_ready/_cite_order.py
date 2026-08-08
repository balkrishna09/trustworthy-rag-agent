"""Compute first-citation order across the document stream and check bib coverage."""
import re, os, sys

def stream(path):
    out = []
    for line in open(path, encoding="utf-8"):
        ls = "" if line.lstrip().startswith("%") else line.split("%")[0]
        m = re.search(r"\\input\{([^}]+)\}", ls)
        if m:
            f = m.group(1)
            if not f.endswith(".tex"):
                f += ".tex"
            if os.path.exists(f):
                out.extend(stream(f))
            continue
        out.append(ls)
    return out

doc = "\n".join(stream("icsea2026_paper.tex"))
order = []
for m in re.finditer(r"\\cite\{([^}]+)\}", doc):
    for k in m.group(1).split(","):
        k = k.strip()
        if k and k not in order:
            order.append(k)

bib = open("sections/08_references.tex", encoding="utf-8").read()
keys_in_bib = re.findall(r"\\bibitem\{([^}]+)\}", bib)
print("cited order (%d):" % len(order))
for i, k in enumerate(order, 1):
    print(f"  [{i}] {k}")
print("uncited bibitems:", [k for k in keys_in_bib if k not in order])
print("cited but missing from bib:", [k for k in order if k not in keys_in_bib])

if "--write" in sys.argv:
    # extract each bibitem block and rewrite in citation order (uncited appended at end)
    blocks = {}
    parts = re.split(r"(?=\\bibitem\{)", bib)
    header = parts[0]
    for p in parts[1:]:
        key = re.match(r"\\bibitem\{([^}]+)\}", p).group(1)
        blocks[key] = p.replace("\\end{thebibliography}", "").rstrip() + "\n"
    ordered = [blocks[k] for k in order if k in blocks]
    ordered += [blocks[k] for k in keys_in_bib if k not in order]
    new = header + "".join(ordered) + "\\end{thebibliography}\n"
    open("sections/08_references.tex", "w", encoding="utf-8", newline="\n").write(new)
    print("WROTE reordered bibliography")
