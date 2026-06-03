# Streamlit Demo Instructions

This demo runs the RAG2riches Streamlit UI locally.

## 1. Install Dependencies

```bash
pip install -e ".[ui,llm,vector]"
```

## 2. Set API Keys (Optional)

Create a `.env` file and add your provider keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
VOYAGE_API_KEY=...
```

## 3. Launch the UI

```bash
streamlit run rag2riches/ui/streamlit_app.py
```

## 4. Try a Sample CSV

Use the sample CSV in `tests/fixtures/speeches_sample.csv`:

- Text column: `speech_text`
- Metadata columns: `party`, `year`, `speaker`
- Cell fields: `party`, `year`

Use the Subset filter section to limit the corpus before comparison (for example,
`year >= 2016` or `company is any of Coca Cola, Pepsi`). Values auto-parse as numbers/bools;
wrap text in quotes to force strings (e.g., `'00123'`).

The embedding model selector includes presets for OpenAI, Google, Anthropic, Voyage AI, and
popular open-weight models. Use Custom if you need a different LiteLLM model identifier or
provider-specific prefixes (ollama/, huggingface/, together/).

## 5. Output Files

The UI writes responses to JSON or CSV based on your selection. Default:

```
output/streamlit_responses.json
```

After running a query, expand each response card to reveal the retrieved chunks and inspect
the grounded context.

