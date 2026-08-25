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

def extract_page_text_no_superscripts(page) -> str:
    """
    Extract text from a page, suppressing superscript spans.
    A span is treated as a superscript if its font size is notably
    smaller than the dominant font size on the page.
    """
    blocks = page.get_text("dict")["blocks"]
    
    # First pass: collect all font sizes to find the dominant (body) size
    font_sizes = []
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span["text"].strip():
                    font_sizes.append(round(span["size"], 1))
    
    if not font_sizes:
        return ""
    
    # Dominant size = most common font size on the page
    from collections import Counter
    dominant_size = Counter(font_sizes).most_common(1)[0][0]
    superscript_threshold = dominant_size * 0.75  # spans < 75% of body = superscript

    # Second pass: build text, skipping superscript spans
    lines_out = []
    for block in blocks:
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                if round(span["size"], 1) >= superscript_threshold:
                    line_text += span["text"]
                # else: silently drop the superscript span
            if line_text.strip():
                line_text = re.sub(r'[†‡§¶\*]+', '', line_text)
                lines_out.append(line_text)

    return "\n".join(lines_out)
    
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
            #text = page.get_text()
            text = extract_page_text_no_superscripts(page)
            if text:
                extracted_text.append(text)
        doc.close()
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}")
        return None

    full_text = "\n".join(extracted_text)
    full_text = re.sub(
       r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])',
       '',
       full_text
    )
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

def strip_repeated_banners(text: str, search_window: int = 20) -> str:
    """
    Remove lines that appear in the first `search_window` lines
    AND recur elsewhere in the document — a reliable signature of
    journal mastheads, running headers, and publisher banners.
    """
    lines = text.split('\n')
    
    # Normalize spaced-character lines before comparison
    # e.g. "S c i e n c e" → "Science"
    def normalize(line: str) -> str:
        collapsed = re.sub(r'(?<=\w) (?=\w)', '', line.strip())
        return collapsed
        
    candidates = set()

    for line in lines[:search_window]:
        stripped = line.strip()
        # Ignore blank lines, page numbers, and very long lines
        if stripped and not stripped.isdigit() and len(stripped) < 80:
            candidates.add(stripped)

    # Keep only candidates that actually recur after the first window
    remainder = '\n'.join(lines[search_window:])
    banners = {c for c in candidates if remainder.count(c) > 1}

    cleaned = [l for l in lines if l.strip() not in banners]
    return '\n'.join(cleaned)

def clean_line_number_interleaving(text: str) -> str:
    # Remove standalone line numbers (1–4 digits on their own line)
    text = re.sub(r'^\s*\d{1,4}\s*$\n?', '', text, flags=re.MULTILINE)
    # Collapse runs of 3+ blank lines down to one, to reduce noise
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text
    
def clean_document_type_labels(text: str) -> str:
    """
    Remove all-caps document-type labels that appear as standalone lines,
    e.g. REVIEW, RESEARCH ARTICLE, EDITORIAL, PERSPECTIVE, COMMENTARY.
    These are structural PDF labels, not titles or content.
    """
    return re.sub(
        r'^\s*(REVIEW|RESEARCH ARTICLE|ORIGINAL ARTICLE|EDITORIAL|'
        r'PERSPECTIVE|COMMENTARY|BRIEF COMMUNICATION|LETTER|OPINION)\s*$\n?',
        '',
        text,
        flags=re.MULTILINE
    )
     
def extract_title_by_author_proximity(text: str) -> str:
    """
    Finds the author line heuristically, then collects ALL contiguous
    non-blank lines immediately above it as the title (handles multi-line titles).
    Handles both "K. N. Thomas" and "Johan S. G. Chua" author formats.
    """
    # Pattern A: initials-first style — "K. N. Thomas1, H. G. Caldwell2"
    author_pattern_a = re.compile(
        r'^(?:[A-Z][A-Za-zÀ-ÿ\.\-]+(?: [A-Z]\.)*[\w\s\.\,]*?)'
        r'(?:,|\s+and\s+)'
        r'(?:[A-Z][A-Za-zÀ-ÿ\.\-]+(?: [A-Z]\.)*[\w\s\.\,]*?)'
        r'[\d\,\*†‡§¶]*\s*$'
    )

    # Pattern B: given-name-first style — "Johan S. G. Chua,1and James A. Evans"
    author_pattern_b = re.compile(
        r'^[A-Z][a-z]+ (?:[A-Z]\. )*[A-Z][A-Za-z\-]+[†‡§¶\*\d\,]*'
        r'(?:,| and )'
        r'[A-Z][a-z]+ (?:[A-Z]\. )*[A-Z][A-Za-z\-]+'
    )

    # Pattern C: long multi-author lines — many given-name-first names, comma-separated
    # Detects lines like "Yu Zheng, Ruyi Cai, Kui Wang, ..." (>3 comma-separated names)
    author_pattern_c = re.compile(
        r'^(?:[A-Z][a-z][\w\-]*(?:\s[A-Z][\w\.\-]*)+ ,\s*){2,}'
    )

    SKIP_PATTERN = re.compile(
        r'^(REVIEW|RESEARCH ARTICLE( SUMMARY)?|ORIGINAL ARTICLE|EDITORIAL|'
        r'PERSPECTIVE|COMMENTARY|BRIEF COMMUNICATION|LETTER|OPINION|'
        r'KEY POINTS?|HIGHLIGHTS?|NEUROSCIENCE|SOCIAL SCIENCES?|PHYSICS|'
        r'CHEMISTRY|BIOLOGY|MEDICINE|ECOLOGY|GENETICS)$',
        re.IGNORECASE
    )

    # Scope to preamble only — title and authors are always in the first ~4000 chars
    lines = [l for l in text[:4000].split('\n') if l.strip()]
    
    print(f"\n--- DEBUG extract_title_by_author_proximity ---")
    print(f"  Total lines in preamble: {len(lines)}")
    for idx, l in enumerate(lines):
        print(f"  [{idx:02d}] {l[:120]}")
        
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Normalize fused superscripts for matching: "Chua,1and" → "Chua, and"
        normalized = re.sub(r'[\d]+(?=[a-z,\s])', '', stripped)

        is_author_line = (
            (len(stripped) < 160
             and stripped.count('.') <= 5
             and (author_pattern_a.match(normalized) or author_pattern_b.match(normalized))
            )
            or
            # Pattern C: long given-name-first author lists
            (stripped.count(',') >= 3
             and author_pattern_b.match(normalized[:60]))  # first 60 chars match name pattern
        )

        if is_author_line:
            # Walk upward; skip any preceding author-continuation lines
            # (lines that are also long and comma-heavy with no sentence structure)
            start = i
            while start > 0:
                prev = lines[start - 1].strip()
                prev_normalized = re.sub(r'[\d]+(?=[a-z,\s])', '', prev)
                # Stop if this line looks like the FIRST author line (pattern B match = given name first)
                if author_pattern_b.match(prev_normalized):
                    break
                if (prev.count(',') >= 2
                        and not re.search(r'[.?!]\s', prev)
                        and not SKIP_PATTERN.match(prev)
                        and not re.search(r'\b(Vol|pp|doi|https?://|Journal of|full article|author affiliation)\b',
                                          prev, re.IGNORECASE)):
                    start -= 1
                else:
                    break

            # Now collect title lines above `start`
            title_lines = []
            for j in range(start - 1, max(start - 8, -1), -1):
                candidate = lines[j].strip()
                if not candidate:
                    break
                if SKIP_PATTERN.match(candidate):
                    break
                if re.search(r'\b(Vol|pp|doi|https?://|Journal of|full article|author affiliation)\b',
                             candidate, re.IGNORECASE):
                    break
                if candidate.endswith(('.', ':', '?', '!')):
                    title_lines.insert(0, candidate)
                    break
                title_lines.insert(0, candidate)

            if title_lines:
                return ' '.join(title_lines)

    return ""
   
def extract_title_fallback(text: str) -> str:
    """
    Last-resort title extraction: return the first non-blank line
    in the preamble that isn't a journal label, banner, or metadata line.
    Used when author-proximity detection fails entirely.
    """
    SKIP_PATTERN = re.compile(
        r'^(REVIEW|RESEARCH ARTICLE( SUMMARY)?|ORIGINAL ARTICLE|EDITORIAL|'
        r'PERSPECTIVE|COMMENTARY|BRIEF COMMUNICATION|LETTER|OPINION|'
        r'KEY POINTS?|HIGHLIGHTS?|NEUROSCIENCE|SOCIAL SCIENCES?)$',
        re.IGNORECASE
    )
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:10]:
        if SKIP_PATTERN.match(line):
            continue
        if re.search(r'\b(Vol|pp\.|doi|https?://|Journal of|^\d+$)\b', line, re.IGNORECASE):
            continue
        if len(line) < 20:   # too short to be a title
            continue
        return line
    return ""
       
def extract_paper_metadata(full_text: str, model_name: str) -> dict:
    """Ask Ollama to extract title, author last names, year, and keywords from the raw text."""
    cleaned_text = strip_repeated_banners(full_text)
    cleaned_text = clean_line_number_interleaving(cleaned_text) 
    cleaned_text = clean_document_type_labels(cleaned_text)   # ← add this
    preamble = cleaned_text[:4000]
    
    # Deterministic title candidate to anchor the LLM
    title_hint = extract_title_by_author_proximity(cleaned_text)
    if not title_hint:
        title_hint = extract_title_fallback(cleaned_text)
    
    hint_instruction = (
        f"The title is most likely: \"{title_hint}\"\n"
        "Use this as the title unless it is clearly wrong "
        "(e.g. it is a journal name, affiliation, or section heading).\n\n"
    ) if title_hint else ""
    
    print("  Extracting paper metadata (title, authors, year, keywords)...")
    response = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": (
                "The following text is extracted from an academic journal PDF. "
                "PDF extraction often places publisher or journal metadata at the very top — "
                "such as a journal name, volume/issue line, page range, or DOI — "
                "before the actual article title appears. "
                "These are NOT the article title. "
                "To locate the title: first find the author names, then take the single line "
                "immediately above them — that exact line is the title. "
                "If there are multiple heading-like lines above the authors, choose only the "
                "one directly adjacent to the author line, not lines further above it. "
                "A section label like 'REVIEW', 'RESEARCH ARTICLE', or a thematic series "
                "heading above the title is NOT the title.\n\n"
                "From the following text, extract:\n"
                "1. The full title of the paper\n"
                "2. The last names of all authors as a comma-separated list. "
                "Strip any superscript affiliation numbers or symbols immediately "
                "following a name (e.g. 'Smith1,2' should be returned as 'Smith').\n"
                "3. The year of publication\n"
                "4. Any explicitly stated keywords (usually listed under a 'Keywords' heading)\n\n"
                "Return ONLY a JSON object with keys: \"title\", \"authors\", \"year\", \"keywords\".\n"
                "Example: {\"title\": \"Deep Learning for Vision\", \"authors\": \"Smith, Jones\", "
                "\"year\": \"2021\", \"keywords\": \"deep learning, image segmentation, CNN\"}\n"
                "For keywords, return only explicitly stated ones — do not infer them.\n"
                "If any field cannot be found, use an empty string.\n\n"
                f"{hint_instruction}"
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

    # Remove repeated Wiley/Journal of Physiology page mastheads and
    # rejoin words split across PDF line breaks.
    full_document_text = re.sub(
        r'(?im)(?:'
        # Wiley copyright masthead
        r'^\s*©[^\n]*\bThe Journal of Physiology\b[^\n]*\n?'
        # Wiley download/terms line
        r'|^\s*\d{6,}[^\n]*\bDownloaded from https?://[^\n]*\n?'
        # Standalone page number, e.g. 3938
        r'|^\s*\d{3,5}\s*$\n?'
        # Repeated author running header, e.g. "H. Fiumelli and others"
        r'|^\s*[A-Z](?:\.\s*[A-Z])?\.\s+[A-Z][A-Za-zÀ-ÿ\'-]+(?:\s+and others)?\s*$\n?'
        # Repeated abbreviated journal/volume running header
        r'|^\s*J\s+Physiol\s+\d+(?:\.\d+)?\s*$\n?'
        # Words split by line wrapping: "neuro-\nplasticity" → "neuroplasticity"
        r'|(?<=[A-Za-z])[-\u2010\u2011]\s*\n\s*(?=[a-z])'
        r')',
        '',
        full_document_text
    )

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
    # Match superscript affiliation markers: digits and commas between them,
    # but only when they follow a letter and precede a separator or end of string.
    # e.g. "Dempsey1,2" → "Dempsey",  "McCrimmon3" → "McCrimmon"
    paper_authors = re.sub(r'(?<=[A-Za-z])\d+(?:,\d+)*(?=[,\s]|$)', '', paper_authors)

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

            Instructions:
            - Select only labels that are a central topic of the document.
            - If the document is a review, opinion, essay, or perspective paper rather
              than an original research article, include the label 'review' if it is
              in the allowed labels list.
            - Return ONLY a JSON array of exact allowed labels.
              Do not include explanations, confidence ratings, parenthetical notes,
              markdown, or labels not in the allowed list.
            - If no label applies, return [].
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
