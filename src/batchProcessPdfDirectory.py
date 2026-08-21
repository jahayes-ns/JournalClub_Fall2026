#!/usr/bin/env python
import os
import glob
import re
import pymupdf  
import ollama
import json
import markdown

SECTION_PATTERN = re.compile(
    r'^\s*(abstract|introduction|background|methods?|methodology|'
    r'results?|discussion|conclusion|references|acknowledgements?)\s*$',
    re.IGNORECASE
)

# --------------------------------------------------------------------------
# Text extraction (unchanged from your original, with raw text caching)
# --------------------------------------------------------------------------
def extract_full_text(pdf_path: str, basename: str, directory: str) -> str:
    raw_txt_filename = os.path.join(directory, f"{basename}_raw.txt")

    if os.path.exists(raw_txt_filename):
        print(f"  Loading cached raw text for {basename}...")
        with open(raw_txt_filename, "r", encoding="utf-8") as f:
            return f.read()

    print(f"  Extracting text from PDF for {basename}...")
    extracted_text = []
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:
            text = page.get_text()
            if text:
                extracted_text.append(text)
        doc.close()
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}")
        return None

    full_text = "\n".join(extracted_text)
    with open(raw_txt_filename, "w", encoding="utf-8") as f:
        f.write(full_text)
    return full_text


# --------------------------------------------------------------------------
# Section splitting
# --------------------------------------------------------------------------
def split_into_sections(full_text: str) -> dict:
    sections = {}
    current_section = "preamble"
    sections[current_section] = []

    for line in full_text.split('\n'):
        if SECTION_PATTERN.match(line):
            current_section = line.strip().lower()
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    return {k: '\n'.join(v).strip() for k, v in sections.items() if v}


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str, max_tokens: int = 3800, chars_per_token: int = 4) -> list:
    max_chars = max_tokens * chars_per_token
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]


# --------------------------------------------------------------------------
# Summarization helpers (using ollama.chat to match your existing style)
# --------------------------------------------------------------------------
def summarize_chunk(chunk: str, section_name: str, model_name: str) -> str:
    response = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": (
                f"You are summarizing the '{section_name}' section of a scientific paper. "
                f"Be concise and preserve key findings, methods, and data.\n\n{chunk}"
            )
        }]
    )
    return response['message']['content'].strip()


def summarize_section(section_name: str, section_text: str, model_name: str) -> str:
    chunks = chunk_text(section_text)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"  Summarizing '{section_name}' chunk {i+1}/{len(chunks)}...")
        chunk_summaries.append(summarize_chunk(chunk, section_name, model_name))

    if len(chunk_summaries) > 1:
        combined = '\n\n'.join(chunk_summaries)
        print(f"  Merging '{section_name}' chunk summaries...")
        return summarize_chunk(combined, f"{section_name} (combined)", model_name)

    return chunk_summaries[0]


def build_meta_summary(section_summaries: dict, model_name: str) -> str:
    combined = '\n\n'.join(
        f"### {k.upper()}\n{v}" for k, v in section_summaries.items()
    )
    print("  Generating final meta-summary...")
    response = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": (
                "You are given section-by-section summaries of a scientific paper. "
                "Write a single cohesive summary covering: research question, methods, "
                "key findings, and conclusions. Be concise but complete.\n\n"
                f"{combined}"
            )
        }]
    )
    return response['message']['content'].strip()

def extract_paper_metadata(full_text: str, model_name: str) -> dict:
    """Ask Ollama to extract title, author last names, year, and keywords from the raw text."""
    preamble = full_text[:4000]
    print("  Extracting paper metadata (title, authors, year, keywords)...")
    response = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": (
                "From the following text, extract:\n"
                "1. The full title of the paper\n"
                "2. The last names of all authors as a comma-separated list\n"
                "3. The year of publication\n"
                "4. Any explicitly stated keywords (usually listed under a 'Keywords' heading)\n\n"
                "Return ONLY a JSON object with keys: \"title\", \"authors\", \"year\", \"keywords\".\n"
                "Example: {\"title\": \"Deep Learning for Vision\", \"authors\": \"Smith, Jones\", "
                "\"year\": \"2021\", \"keywords\": \"deep learning, image segmentation, CNN\"}\n"
                "For keywords, return only explicitly stated ones — do not infer them.\n"
                "If any field cannot be found, use an empty string.\n\n"
                f"{preamble}"
            )
        }],
        options={"temperature": 0}
    )
    raw = response['message']['content'].strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\n?```$', '', raw)
    try:
        metadata = json.loads(raw)
        if not isinstance(metadata, dict):
            raise ValueError("Expected a JSON object")
    except (json.JSONDecodeError, ValueError):
        metadata = {"title": "", "authors": "", "year": "", "keywords": ""}
    return metadata
    
# --------------------------------------------------------------------------
# Main analysis function (replaces your original analyze_pdf_locally)
# --------------------------------------------------------------------------
def analyze_pdf_locally(pdf_path: str, topics_path: str, model_name: str = "mistral:latest"):
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    directory = os.path.dirname(pdf_path)

    # 1. Extract (or load cached) raw text
    full_document_text = extract_full_text(pdf_path, basename, directory)
    if not full_document_text:
        return None

    # 2. Split into sections and summarize each one
    sections = split_into_sections(full_document_text)
    section_summaries = {}
    for section_name, section_text in sections.items():
        if section_name == "references":
            continue  # Skip references
        section_summaries[section_name] = summarize_section(section_name, section_text, model_name)

    # 3. Build final meta-summary
    summary_text = build_meta_summary(section_summaries, model_name)

    # Extract title, authors, year from preamble
    metadata = extract_paper_metadata(full_document_text, model_name)
    paper_title   = metadata.get("title", "").strip()
    paper_authors = metadata.get("authors", "").strip()
    paper_year    = metadata.get("year", "").strip()
    paper_keywords = metadata.get("keywords", "").strip()

    # Save metadata to JSON for reuse by other scripts
    meta_filename = os.path.join(directory, f"{basename}_meta.json")
    with open(meta_filename, "w", encoding="utf-8") as meta_file:
        json.dump({
            "title":    paper_title,
            "authors":  paper_authors,
            "year":     paper_year,
            "keywords": paper_keywords,
            "basename": basename,
            "pdf_path": pdf_path
        }, meta_file, indent=4)
    print(f"  Metadata saved to {basename}_meta.json")
    
    # Build the headline block for reuse in both .md and .html
    if paper_title:
        md_headline = f"# {paper_title}\n\n"
    else:
        md_headline = f"# {basename}\n\n"
    if paper_authors or paper_year:
        byline_parts = []
        if paper_authors:
            byline_parts.append(paper_authors)
        if paper_year:
            byline_parts.append(paper_year)
        md_headline += f"**{' | '.join(byline_parts)}**\n\n---\n\n"
    if paper_keywords:
        md_headline += f"**Keywords:** {paper_keywords}\n\n---\n\n"
                
    # 4. Read topics and classify using the meta-summary (unchanged from your original)
    with open(topics_path, 'r', encoding="utf-8") as f:
        topics_list = f.read().strip()

    print(f"  Classifying topics for {basename}...")
    allowed_topics = [
        line.strip()
        for line in topics_list.splitlines()
        if line.strip()
    ]

    classification_response = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": f"""Classify the document using the allowed labels below.

            Allowed labels:
            {chr(10).join(allowed_topics)}

            Document summary:
            {summary_text}

            Return ONLY a JSON array of exact allowed labels.
            Do not include explanations, confidence ratings, parenthetical notes, markdown,
            or labels not in the allowed list.
            Select only labels that are a central topic of the document.
            If no label applies, return [].
            """
        }],
        options={"temperature": 0}
    )

    raw_output = classification_response["message"]["content"].strip()

    try:
        proposed_topics = json.loads(raw_output)
        if not isinstance(proposed_topics, list):
            raise ValueError("Expected a JSON list")
    except (json.JSONDecodeError, ValueError):
        proposed_topics = []

    # Enforcement: discard invalid labels even if the model ignores the prompt.
    topics = [
        topic for topic in proposed_topics
        if isinstance(topic, str) and topic in allowed_topics
    ]

    classification_text = ", ".join(topics)

    # 5. Write section summaries + meta-summary to Markdown (extended format)
    summary_filename = os.path.join(directory, f"{basename}.md")
    with open(summary_filename, "w", encoding="utf-8") as md_file:
        md_file.write(md_headline)                          # <-- headline first
        md_file.write("## Final Summary\n\n")
        md_file.write(summary_text)
        md_file.write("\n\n---\n\n## Section Summaries\n\n")
        for section_name, section_summary in section_summaries.items():
            md_file.write(f"### {section_name.upper()}\n\n{section_summary}\n\n")

    # 6. Write topics file (unchanged)
    topics_filename = os.path.join(directory, f"{basename}_topics.txt")
    with open(topics_filename, "w", encoding="utf-8") as topics_file:
        topics_file.write(classification_text)

    return {
        "filename": pdf_path,
        "summary_file": summary_filename,
        "topics_file": topics_filename,
        "topics": [topic.strip() for topic in classification_text.split(',')]
    }


# --------------------------------------------------------------------------
# Directory processor (unchanged from your original)
# --------------------------------------------------------------------------
def process_directory(directory_path: str, topics_path: str, model_name: str = "mistral:latest"):
    search_pattern = os.path.join(directory_path, "*.pdf")
    pdf_files = glob.glob(search_pattern)

    if not pdf_files:
        print(f"No PDFs found in {directory_path}")
        return

    for pdf_path in pdf_files:
        base_name = os.path.splitext(pdf_path)[0]
        expected_md = base_name + ".md"
        expected_html = base_name + ".html"

        # If HTML already exists, skip entirely
        if os.path.exists(expected_html):
            print(f"Skipping: {pdf_path} (HTML already exists)")
            continue

        # If .md exists but .html doesn't, convert directly without calling Ollama
        if os.path.exists(expected_md):
            print(f"Regenerating HTML from existing .md for: {pdf_path}")
            with open(expected_md, "r", encoding="utf-8") as f:
                md_content = f.read()
            html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
            with open(expected_html, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"  Done. HTML saved to: {expected_html}")
            continue
            
        # Neither exists — full processing required
        print(f"\nProcessing: {pdf_path}...")
        result = analyze_pdf_locally(pdf_path, topics_path, model_name)
        if not result:
            print(f"  Failed to process {pdf_path}, skipping.")
            continue

        md_filepath = result["summary_file"]
        with open(md_filepath, "r", encoding="utf-8") as f:
            md_content = f.read()

        html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])

        html_filepath = os.path.splitext(md_filepath)[0] + ".html"
        with open(html_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"  Done. HTML saved to: {html_filepath}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    process_directory("./pdfs", "topics.txt", model_name="mistral:latest")
