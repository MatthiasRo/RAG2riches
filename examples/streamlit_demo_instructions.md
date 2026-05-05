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

## 5. Output Files

The UI writes responses to JSON or CSV based on your selection. Default:

```
output/streamlit_responses.json
```

