LIC Virtual Agent (RAG + Gemini free)

Overview
- Small demo of a virtual LIC premium-reminder agent.
- Uses a basic RAG pipeline (TF-IDF retrieval) and a Gemini client wrapper.
- PII (names, phone, email) is redacted before sending content to the LLM; local mapping inserts PII when "sending" messages.

Files
- data/policyholders.csv : pseudo policyholder data
- src/rag.py : simple TF-IDF-based retrieval/indexing
- src/privacy.py : PII redaction and mapping
- src/gemini_client.py : wrapper to call Gemini (if configured) or fallback
- src/agent.py : agent logic and CLI multi-turn demo
- run_demo.py : small runner to talk to a chosen policyholder

Usage (quick):
1) Create a venv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) (Optional) Configure Gemini free API key in `.env`:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

3) Run the demo:

```bash
python run_demo.py
```

Notes on privacy
- The agent redacts PII before including policyholder details in prompts.
- Real contact details remain in local mapping and never go to the LLM.

Limitations
- This is a demo scaffold; replace the retrieval/LLM pieces for production.
