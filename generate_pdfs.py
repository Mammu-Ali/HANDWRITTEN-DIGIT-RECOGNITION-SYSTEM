"""
PDF Generation Script for Handwritten Digit Recognition Project
Generates:
1. PROJECT_REPORT.pdf (Full 13-Page Technical Report)
2. SRS.pdf (2-Page Software Requirements Specification)

Uses Microsoft Edge Headless with strict CSS page-break specifications,
vector diagrams, and publication-ready typography.
"""

import os
import re
import subprocess
import sys
import markdown_it

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PROJECT_DIR = r"D:\RealTimeWork\Handwritten Digit Recognition"
MD_REPORT_PATH = os.path.join(PROJECT_DIR, "PROJECT_REPORT.md")
MD_SRS_PATH = os.path.join(PROJECT_DIR, "SRS.md")
PDF_REPORT_PATH = os.path.join(PROJECT_DIR, "PROJECT_REPORT.pdf")
PDF_SRS_PATH = os.path.join(PROJECT_DIR, "SRS.pdf")
HTML_REPORT_PATH = os.path.join(PROJECT_DIR, "project_report_print.html")
HTML_SRS_PATH = os.path.join(PROJECT_DIR, "srs_print.html")

md_parser = markdown_it.MarkdownIt("commonmark", {"html": True, "breaks": True}).enable("table")

SHARED_CSS = """
@page {
    size: A4 portrait;
    margin: 12mm 14mm 12mm 14mm;
}

* {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
    font-size: 8.5pt;
    line-height: 1.4;
}

.page {
    page-break-after: always;
    break-after: page;
    height: 270mm;
    max-height: 270mm;
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
    padding: 2mm 0;
}

.page:last-child {
    page-break-after: auto;
    break-after: auto;
}

.page-header {
    flex: 0 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1.5px solid #0284c7;
    padding-bottom: 4px;
    margin-bottom: 8px;
    font-size: 7.5pt;
    font-weight: 600;
    color: #0369a1;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.page-footer {
    flex: 0 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #cbd5e1;
    padding-top: 4px;
    margin-top: 6px;
    font-size: 7.5pt;
    color: #64748b;
}

.page-content {
    flex: 1 1 auto;
    overflow: hidden;
}

h1.page-title {
    font-size: 13pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 3px;
    margin: 0 0 8px 0;
    letter-spacing: -0.3px;
}

h2 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #1e3a8a;
    margin: 8px 0 4px 0;
    border-left: 3px solid #3b82f6;
    padding-left: 6px;
}

h3 {
    font-size: 9pt;
    font-weight: 700;
    color: #1e293b;
    margin: 6px 0 3px 0;
}

p {
    margin: 0 0 5px 0;
    color: #334155;
    text-align: justify;
}

ul, ol {
    margin: 0 0 6px 0;
    padding-left: 18px;
}

li {
    margin-bottom: 2px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0 8px 0;
    font-size: 7.8pt;
}

th {
    background: #1e40af;
    color: #ffffff;
    font-weight: 600;
    padding: 4px 6px;
    text-align: left;
    border: 1px solid #1e3a8a;
}

td {
    padding: 3.5px 6px;
    border: 1px solid #cbd5e1;
    color: #1e293b;
}

tr:nth-child(even) td {
    background-color: #f8fafc;
}

pre {
    background: #0f172a;
    color: #f8fafc;
    padding: 6px 8px;
    border-radius: 4px;
    font-family: Consolas, "Liberation Mono", Courier, monospace;
    font-size: 7.5pt;
    line-height: 1.25;
    margin: 4px 0 6px 0;
    overflow: hidden;
    white-space: pre-wrap;
}

code {
    font-family: Consolas, "Liberation Mono", Courier, monospace;
    font-size: 8pt;
    background: #e2e8f0;
    color: #0f172a;
    padding: 1px 3px;
    border-radius: 3px;
}

.math-display {
    background: #f1f5f9;
    border-left: 3px solid #6366f1;
    padding: 4px 8px;
    margin: 4px 0 6px 0;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 8.8pt;
    color: #1e1b4b;
    font-style: italic;
}

.cover-card {
    border: 2px solid #1e3a8a;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    margin: 10px 0 16px 0;
}

.cover-title {
    font-size: 16pt;
    font-weight: 800;
    color: #1e3a8a;
    margin-bottom: 8px;
    line-height: 1.2;
}

.cover-subtitle {
    font-size: 10pt;
    font-weight: 600;
    color: #0369a1;
    margin-bottom: 12px;
}

.cover-meta {
    font-size: 8pt;
    color: #475569;
    margin: 3px 0;
}

.diagram-card {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
    margin: 6px 0;
    font-family: Consolas, monospace;
    font-size: 7.5pt;
}

.diagram-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    margin: 4px 0;
    flex-wrap: wrap;
}

.diagram-node {
    background: #ffffff;
    border: 1.5px solid #2563eb;
    border-radius: 4px;
    padding: 4px 6px;
    text-align: center;
    font-weight: 600;
    color: #1e40af;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.diagram-node.dark {
    background: #1e40af;
    color: #ffffff;
    border-color: #1e3a8a;
}

.diagram-arrow {
    color: #64748b;
    font-weight: bold;
    font-size: 10pt;
}

.badge {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 7pt;
    font-weight: 600;
}
.badge-blue { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
.badge-green { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.badge-orange { background: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; }

.highlight-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #2563eb;
    padding: 6px 10px;
    border-radius: 4px;
    margin: 6px 0;
    font-size: 8pt;
}
"""

def clean_latex_and_mermaid(text: str) -> str:
    """Transform LaTeX math blocks and mermaid blocks into clean HTML."""
    # LaTeX display math $$...$$
    def replace_math_display(match):
        formula = match.group(1).strip()
        # Common replacements for presentation
        formula = formula.replace(r"\sigma", "&sigma;")
        formula = formula.replace(r"\mathbf{z}", "<b>z</b>")
        formula = formula.replace(r"\mathbf{I}", "<b>I</b>")
        formula = formula.replace(r"\mathbf{K}", "<b>K</b>")
        formula = formula.replace(r"\mathcal{L}", "<b><i>L</i></b>")
        formula = formula.replace(r"\sum", "&sum;")
        formula = formula.replace(r"\cdot", "&middot;")
        formula = formula.replace(r"\max", "max")
        formula = formula.replace(r"\arg\max", "argmax")
        formula = formula.replace(r"\forall", "&forall;")
        formula = formula.replace(r"\in", "&isin;")
        formula = formula.replace(r"\mathbb{R}", "&reals;")
        formula = formula.replace(r"\mu", "&mu;")
        formula = formula.replace(r"\alpha", "&alpha;")
        formula = formula.replace(r"\beta", "&beta;")
        formula = formula.replace(r"\epsilon", "&epsilon;")
        formula = formula.replace(r"\Delta", "&Delta;")
        formula = formula.replace(r"\le", "&le;")
        formula = formula.replace(r"\ge", "&ge;")
        formula = formula.replace(r"\approx", "&asymp;")
        formula = formula.replace(r"\to", "&rarr;")
        formula = formula.replace(r"\times", "&times;")
        formula = formula.replace(r"\frac", "")
        formula = formula.replace(r"\text", "")
        formula = formula.replace(r"\left", "")
        formula = formula.replace(r"\right", "")
        formula = formula.replace("{", "(").replace("}", ")")
        return f'<div class="math-display">{formula}</div>'

    text = re.sub(r"\$\$(.*?)\$\$", replace_math_display, text, flags=re.DOTALL)

    # Inline math $...$
    def replace_math_inline(match):
        formula = match.group(1).strip()
        formula = formula.replace(r"\times", "&times;")
        formula = formula.replace(r"\in", "&isin;")
        formula = formula.replace(r"\mathbb{R}", "&reals;")
        formula = formula.replace(r"\le", "&le;")
        formula = formula.replace(r"\ge", "&ge;")
        formula = formula.replace(r"\mu", "&mu;")
        formula = formula.replace(r"\sigma", "&sigma;")
        formula = formula.replace(r"\alpha", "&alpha;")
        formula = formula.replace(r"\beta", "&beta;")
        formula = formula.replace(r"\Delta", "&Delta;")
        formula = formula.replace(r"\to", "&rarr;")
        formula = formula.replace(r"\bar{x}", "x&#772;")
        formula = formula.replace(r"\bar{y}", "y&#772;")
        return f'<i style="font-family: serif;">{formula}</i>'

    text = re.sub(r"\$(.*?)\$", replace_math_inline, text)

    # Replace mermaid diagrams with clean structural diagrams
    def replace_mermaid(match):
        content = match.group(1)
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        nodes = []
        for line in lines:
            if "-->" in line:
                parts = line.split("-->")
                for p in parts:
                    clean = re.sub(r'[\[\]\(\)\"\']', '', p).replace("<br>", " - ").strip()
                    if clean and clean not in nodes:
                        nodes.append(clean)
        if nodes:
            cards_html = []
            for i, n in enumerate(nodes[:6]):
                cards_html.append(f'<div class="diagram-node">{n}</div>')
                if i < len(nodes[:6]) - 1:
                    cards_html.append('<div class="diagram-arrow">&rarr;</div>')
            return f'<div class="diagram-card"><div class="diagram-row">{" ".join(cards_html)}</div></div>'
        return f'<pre>{content}</pre>'

    text = re.sub(r"```mermaid(.*?)```", replace_mermaid, text, flags=re.DOTALL)
    return text


def build_html_report():
    """Parse PROJECT_REPORT.md into 13 distinct printable pages."""
    with open(MD_REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into 13 pages
    # Patterns: # [PAGE X]
    page_splits = re.split(r"#\s*\[PAGE\s+(\d+)\]", content)
    # page_splits[0] is preamble/title, then page_num, page_content, page_num, page_content...
    
    pages = []
    if len(page_splits) > 1:
        for i in range(1, len(page_splits), 2):
            p_num = int(page_splits[i])
            p_text = page_splits[i+1].strip()
            pages.append((p_num, p_text))
    else:
        # Fallback split
        pages.append((1, content))

    html_pages = []
    total_pages = len(pages)

    for p_num, p_text in pages:
        # Process title and content
        lines = p_text.split("\n")
        page_title = lines[0].replace("#", "").strip()
        body_md = "\n".join(lines[1:])
        body_cleaned = clean_latex_and_mermaid(body_md)
        body_html = md_parser.render(body_cleaned)

        page_html = f"""
        <div class="page" id="report-page-{p_num}">
            <div class="page-header">
                <span>Handwritten Digit Recognition System &bull; Technical Capstone Report</span>
                <span>Section {p_num} &bull; Phase 16</span>
            </div>
            <div class="page-content">
                <h1 class="page-title">{page_title}</h1>
                {body_html}
            </div>
            <div class="page-footer">
                <span>Department of Computer Science &amp; Engineering</span>
                <span>Page {p_num} of {total_pages}</span>
            </div>
        </div>
        """
        html_pages.append(page_html)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Handwritten Digit Recognition — Project Report</title>
    <style>
        {SHARED_CSS}
    </style>
</head>
<body>
    {"".join(html_pages)}
</body>
</html>
"""
    with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated HTML report at: {HTML_REPORT_PATH}")
    return HTML_REPORT_PATH


def build_html_srs():
    """Parse SRS.md into 2 distinct printable pages."""
    with open(MD_SRS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    page_splits = re.split(r"#\s*\[PAGE\s+(\d+)\]", content)
    pages = []
    if len(page_splits) > 1:
        for i in range(1, len(page_splits), 2):
            p_num = int(page_splits[i])
            p_text = page_splits[i+1].strip()
            pages.append((p_num, p_text))
    else:
        pages.append((1, content))

    html_pages = []
    total_pages = len(pages)

    for p_num, p_text in pages:
        lines = p_text.split("\n")
        page_title = lines[0].replace("#", "").strip()
        body_md = "\n".join(lines[1:])
        body_cleaned = clean_latex_and_mermaid(body_md)
        body_html = md_parser.render(body_cleaned)

        page_html = f"""
        <div class="page" id="srs-page-{p_num}">
            <div class="page-header">
                <span>Software Requirements Specification (SRS) &bull; IEEE Std 830-1998</span>
                <span>Baseline v1.0.0</span>
            </div>
            <div class="page-content">
                <h1 class="page-title">{page_title}</h1>
                {body_html}
            </div>
            <div class="page-footer">
                <span>System Engineering &amp; ML Architecture Specification</span>
                <span>Page {p_num} of {total_pages}</span>
            </div>
        </div>
        """
        html_pages.append(page_html)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Software Requirements Specification (SRS)</title>
    <style>
        {SHARED_CSS}
    </style>
</head>
<body>
    {"".join(html_pages)}
</body>
</html>
"""
    with open(HTML_SRS_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated HTML SRS at: {HTML_SRS_PATH}")
    return HTML_SRS_PATH


def convert_html_to_pdf(html_path: str, pdf_path: str):
    """Call Edge headless to render HTML to PDF."""
    print(f"Converting {html_path} -> {pdf_path} via Edge...")
    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        print(f"SUCCESS: Generated {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    else:
        print(f"FAILED to generate {pdf_path}: {res.stderr}")
        sys.exit(1)


def main():
    print("=== Generating Production PDF Documents ===")
    html_report = build_html_report()
    convert_html_to_pdf(html_report, PDF_REPORT_PATH)

    html_srs = build_html_srs()
    convert_html_to_pdf(html_srs, PDF_SRS_PATH)
    print("=== All PDF Documents Successfully Generated ===")


if __name__ == "__main__":
    main()
