## Installation
conda create --name llm
conda activate llm
conda install ollama 
conda install conda-forge::ollama-python
conda install pymupdf
pip install markdown

## Running
### process the pdfs in ./pdfs
> time python batchProcessPdfDirectory.py

# collate the summaries by topic
> python collateSummaries.py
- outputs ./pdfs/index.html and *.html in ./pdfs/html
