import os, sys
import glob
import json
import shutil
import ollama
import re
from urllib.parse import quote

def topic_anchor(topic):
    return "topic-" + quote(topic, safe="")

def extract_doi(raw_txt_path: str) -> str:
    """Return the first DOI found in a raw-text file as a doi.org URL."""
    doi_pattern = re.compile(
        r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)',
        re.IGNORECASE
    )

    try:
        with open(raw_txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None

    match = doi_pattern.search(text)
    if not match:
        return None

    doi = match.group(1).rstrip(".,;:)]}")
    return f"https://doi.org/{doi}"

def load_paper_metadata(directory_path: str, basename: str) -> dict:
    """Load _meta.json for a paper if it exists, otherwise return empty fields."""
    meta_path = os.path.join(directory_path, f"{basename}_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"title": "", "authors": "", "year": "", "keywords": ""}

def format_entry_html(file: str, summary_text: str, meta: dict) -> str:
    """Build a rich list item with title, authors/year, summary, and keywords."""
    title    = meta.get("title", "").strip()
    authors  = meta.get("authors", "").strip()
    year     = meta.get("year", "").strip()
    keywords = meta.get("keywords", "").strip()

    # Fall back to filename if no title extracted
    display_title = title if title else file

    byline_parts = []
    if authors:
        byline_parts.append(authors)
    if year:
        byline_parts.append(year)
    byline = " | ".join(byline_parts)

    html = f'<li>\n'
    html += f'  <a href="html/{file}"><strong>{display_title}</strong></a><br/>\n'
    if byline:
        html += f'  <em>{byline}</em><br/>\n'
    if summary_text:
        html += f'  {summary_text}<br/>\n'
    if keywords:
        html += f'  <small><strong>Keywords:</strong> {keywords}</small>\n'
    html += f'</li>\n'
    return html

def generate_topic_index(directory_path: str, master_topics_path: str, output_filename: str = "index.html", model_name: str = "mistral:latest"):
    # Read the master list of topics
    with open(master_topics_path, 'r', encoding="utf-8") as f:
        master_topics = [line.strip() for line in f if line.strip()]

    topic_mapping = {topic: [] for topic in master_topics}
    all_html_files = set()

    # Create html/ subdirectory if it doesn't exist
    html_subdir = os.path.join(directory_path, "html")
    os.makedirs(html_subdir, exist_ok=True)

    # Load cached summaries
    cache_filename = os.path.join(directory_path, "summaries_cache.json")
    file_summaries = {}
    if os.path.exists(cache_filename):
        with open(cache_filename, "r", encoding="utf-8") as f:
            file_summaries = json.load(f)
            print(f"Loaded {len(file_summaries)} cached summaries from disk.")

    search_pattern = os.path.join(directory_path, "*_topics.txt")
    topic_files = glob.glob(search_pattern)

    # Store metadata keyed by html_filename for use when building index.html
    file_metadata = {}

    i = 0
    for topic_file in topic_files:
        basename = os.path.basename(topic_file).replace('_topics.txt', '')
        html_filename = f"{basename}.html"
        html_path = os.path.join(directory_path, html_filename)
        pdf_filename = f"{basename}.pdf"
        md_filename = f"{basename}.md"
        md_path = os.path.join(directory_path, md_filename)

        if not os.path.exists(html_path):
            continue

        all_html_files.add(html_filename)

        # Load _meta.json and stash for index building
        meta = load_paper_metadata(directory_path, basename)
        file_metadata[html_filename] = meta

        # Generate one-sentence summary if not cached
        if html_filename not in file_summaries:
            one_sentence_summary = ""
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
                print(f"Generating new one-sentence summary for: {md_filename}...")
                try:
                    summary_response = ollama.chat(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": f"Summarize the following text in exactly one sentence:\n\n{md_content}"
                        }]
                    )
                    one_sentence_summary = summary_response['message']['content'].strip()
                except Exception as e:
                    print(f"Error generating summary for {md_filename}: {e}")
            file_summaries[html_filename] = one_sentence_summary

        # Build header for individual HTML file with DOI/PDF links
        raw_txt_path = os.path.join(directory_path, f"{basename}_raw.txt")
        doi_url = extract_doi(raw_txt_path)

        if doi_url:
            link_html = f"<a href='{doi_url}' target='_blank'>View Original (DOI)</a><br/>"
            link_html += f"<a href='../{pdf_filename}' target='_blank'>View Original (Local)</a>"
        else:
            link_html = f"<a href='../{pdf_filename}' target='_blank'>View Original (Local)</a>"

        header_html = f"<h3><a href='https://drive.google.com/drive/folders/1Hs9ifhlZb5HSJ5r51T7jK_gQyfR3npIT'>{basename}.pdf</a></h3>\n<p>{link_html}</p>\n<hr>\n"

        # Prepend header to the source HTML if not already done
        with open(html_path, 'r', encoding="utf-8") as f:
            existing_html = f.read()

        if "View Original" not in existing_html:
            with open(html_path, 'w', encoding="utf-8") as f:
                f.write(header_html + existing_html)

        # Copy HTML file into html/ subdirectory
        dest_path = os.path.join(html_subdir, html_filename)
        shutil.copy2(html_path, dest_path)
        print(f"  Copied {html_filename} -> html/{html_filename}")

        # Read topics
        with open(topic_file, 'r', encoding="utf-8") as f:
            assigned_topics_text = f.read().strip()

        assigned_topics = [t.strip() for t in assigned_topics_text.split(',')]

        for topic in assigned_topics:
            if topic in topic_mapping:
                topic_mapping[topic].append(html_filename)
            elif topic:
                if topic not in topic_mapping:
                    topic_mapping[topic] = []
                topic_mapping[topic].append(html_filename)
        i += 1

    # Save updated summaries cache
    with open(cache_filename, "w", encoding="utf-8") as f:
        json.dump(file_summaries, f, indent=4)

    # Build index.html — all links point to html/filename.html
    html_content = "<html>\n<head>\n<title>Document Index by Topic</title>\n"
    html_content += "<style>\n"
    html_content += "  body { font-family: Arial, sans-serif; line-height: 1.4; }\n"
    html_content += "  h2 { margin-top: 30px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }\n"
    html_content += "  small { color: #555; }\n"
    html_content += "  li { margin-bottom: 16px; }\n"
    html_content += "</style>\n"
    html_content += "</head>\n<body>\n"

    topics_with_files = [
        topic for topic in sorted(topic_mapping)
        if topic_mapping[topic]
    ]
    html_content += '<div style="margin-top: 20px;"><a href="#all-papers" style="font-size: 1.1em; font-weight: bold;">View All Papers</a></div>\n'
    html_content += "<h2>Topics</h2>\n<ul>\n"
    html_content += '<ul style="column-count: 4; column-gap: 20px; list-style-type: none; padding: 0;">\n'
    for topic in topics_with_files:
        print("topic: " + topic)
        html_content += f'<li style="margin-bottom: 0px;"><a href="#{topic_anchor(topic)}">{topic}</a></li>\n' 
    html_content += "</ul>\n"
  
    html_content += "<h1>Document Index by Topic</h1>\n"
    
    for topic in sorted(topic_mapping.keys()):
        files = topic_mapping[topic]
        if files:
            html_content += f'<h2 id="{topic_anchor(topic)}">{topic}</h2>\n<ul>\n'
            for file in sorted(files, key=lambda f: file_metadata.get(f, {}).get('title', f)):
                summary_text = file_summaries.get(file, "")
                meta = file_metadata.get(file, {})
                html_content += format_entry_html(file, summary_text, meta)
            html_content += "</ul>\n"

    if all_html_files:
        html_content += '<h2 id="all-papers">All</h2>\n<ul>\n'
        for file in sorted(all_html_files, key=lambda f: file_metadata.get(f, {}).get('title', f)):
            summary_text = file_summaries.get(file, "")
            meta = file_metadata.get(file, {})
            html_content += format_entry_html(file, summary_text, meta)
        html_content += "</ul>\n"

    html_content += "</body>\n</html>"

    # Write index.html to the directory root (not inside html/)
    output_path = os.path.join(directory_path, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Index HTML successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_topic_index("./pdfs", "topics.txt")
