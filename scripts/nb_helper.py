"""
Script to generate the comprehensive, Kaggle-compatible Jupyter notebooks
for all three ecological formulations.
"""

import json
import os

def create_code_cell(source_code):
    lines = source_code.strip().split('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    }

def create_markdown_cell(markdown_text):
    lines = markdown_text.strip().split('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    }

def save_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Saved notebook with {len(cells)} cells to: {filepath}")
