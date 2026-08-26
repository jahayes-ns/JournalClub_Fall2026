# Journal Club PDF Summarizer

Batch-processes a directory of PDFs using a local LLM (Mistral via Ollama),
generating per-paper summaries and a collated HTML index organized by topic.

---

## Requirements

- [Ollama](https://ollama.com) (installed at the system level — see below)
- [Miniconda](https://docs.conda.io/projects/conda/en/stable/index.html)
- Python packages: `ollama`, `pymupdf`, `markdown`

---

## Installation

### 1. Install Ollama (system-level, outside Conda)

Ollama must be installed at the system level so its server binary is available.

- **macOS / Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **Windows:** Download the installer from https://ollama.com/download

Verify the installation:
```bash
ollama --version
```

> ⚠️ Do **not** install Ollama via `conda-forge::ollama-python` — that package
> omits the required server binary and will cause a `llama-server not found` error.

---

### 2. Pull the Mistral model

```bash
ollama pull mistral
```

Verify the model downloaded successfully:
```bash
ollama list
```

---

### 3. Create the Conda Python environment

```bash
conda create --name llm python=3.11
conda activate llm
pip install ollama pymupdf markdown
```

---

### 4. Start the Ollama server

On **macOS** and **Windows**, the Ollama app may start automatically.
On **Linux**, start it manually:

```bash
ollama serve &
```

Verify it is running:
```bash
curl http://localhost:11434
```

---

## Running

### Activate the environment
```bash
conda activate llm
```

### Process the PDFs in `./pdfs`
```bash
python batchProcessPdfDirectory.py
```

### Collate the summaries by topic
```bash
python collateSummaries.py
```

- Outputs `./pdfs/index.html` and per-paper `*.html` files in `./pdfs/html`
