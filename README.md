# -AI-powered-question-answering-agent

Create & Activate Virtual Environment----------------
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Install Dependencies--------------

pip install requests,
beautifulsoup4,
transformers,
sentence-transformers,
scikit-learn,
numpy,
tqdm


Run the agent on a documentation page:
python ai.py https://www.w3schools.com/html/html_intro.asp

Then, ask questions like:
> What is HTML?
> Who created HTML?
> What can HTML be used for?


⚠Known Limitations ---------------------
Basic crawler; does not handle JS-heavy or dynamic content

Handles only simple HTML pages

Slower with larger documents

Limited evaluation on multi-source documents
