import os
import sys
import asyncio
import nbformat
from nbclient import NotebookClient

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def run_nb(nb_name):
    nb_path = os.path.join(r'c:\Users\ARPIT SINGH\OneDrive\Desktop\ISLE_ROYALE\isle_royale_ecosystem\notebooks', nb_name)
    print(f"\n{'='*70}\nExecuting {nb_name}...\n{'='*70}")
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name='python3',
        resources={'metadata': {'path': os.path.dirname(nb_path)}}
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"SUCCESS: {nb_name} executed with 0 errors and all outputs saved!")

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '01_standard_lotka_volterra.ipynb'
    run_nb(target)
