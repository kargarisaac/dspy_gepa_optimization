import os
from langchain_community.document_loaders import PyPDFLoader

DOCS_DIR = os.path.join(os.path.dirname(__file__), 'docs')

for filename in os.listdir(DOCS_DIR):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(DOCS_DIR, filename)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Concatenate all page contents
        md_content = '\n\n'.join(page.page_content for page in pages)
        md_filename = os.path.splitext(filename)[0] + '.md'
        md_path = os.path.join(DOCS_DIR, md_filename)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Converted {filename} to {md_filename}")
