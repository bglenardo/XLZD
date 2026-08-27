import sys
import time
import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": "."}})

t0 = time.time()
try:
    client.execute()
finally:
    nbformat.write(nb, path)
    print(f"Executed in {time.time()-t0:.1f}s")

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            print(f"--- cell {i} stdout ---")
            print(out["text"])
        elif out.get("output_type") == "error":
            print(f"--- cell {i} ERROR ---")
            print("\n".join(out.get("traceback", [])))
