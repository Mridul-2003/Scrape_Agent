# Web Data Extractor

A Streamlit application that uses a language model and a web scraping tool to extract structured information from a webpage. Enter a URL, describe the fields you need, choose an output format, and review the result.

## Features

- Extract user-selected information as readable text, JSON, or a Markdown table.
- Require a scraping tool call for the starting URL and allow the model to follow relevant pagination links.
- Restrict pagination requests to the starting URL's host and enforce a configurable page limit.
- Show run logs and optionally include scraped page content in those logs.
- Select Gemini, Groq, or Ollama as the model provider.

## Requirements

- Python 3.12 or newer
- An API key for Gemini or Groq, or an Ollama setup that can serve the configured model
- Network access to the target website

## Installation

Clone the repository and enter the project directory. Then install dependencies with `uv`:

```bash
uv sync
```

Or use `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root. Set `LLM_PROVIDER` to `gemini`, `groq`, or `ollama`:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

Only the key for the selected hosted provider is needed. For `ollama`, configure Ollama separately and make sure the model configured in `scrap.py` is available. The current provider defaults to Ollama when `LLM_PROVIDER` is missing or does not match `gemini` or `groq`.

Keep `.env` private and do not commit API keys.

## Run the app

From the project root, start Streamlit:

```bash
streamlit run app.py
```

In the app:

1. Enter the webpage URL.
2. Describe the data to extract, such as “Extract each pet's name, breed, and location.”
3. Choose `text`, `json`, or `table` output.
4. Set the maximum number of pages to inspect.
5. Select **Extract** and review the run logs and result.

The optional page-text logging checkbox displays scraped content in the app's run logs. Use it when diagnosing extraction problems; page contents may be long.

## How it works

The application creates a model based on `LLM_PROVIDER`, supplies a `scrape_page` tool, and instructs the model to call that tool for the starting URL. The tool fetches the page and parses it with Beautiful Soup. The model may request additional pages when it finds relevant pagination links. Requests are limited to the original host and the page limit selected in the UI.

The model returns the requested data in the selected format. JSON output is validated and, when needed, the model is asked to return valid JSON again.

## Configuration reference

| Setting | Purpose |
| --- | --- |
| `LLM_PROVIDER` | Provider selection: `gemini`, `groq`, or `ollama` |
| `GOOGLE_API_KEY` | Credentials for Gemini |
| `GROQ_API_KEY` | Credentials for Groq |
| Page limit | Maximum pages the agent can fetch during a run |
| Show page text in logs | Displays fetched page content in the run log |

The provider model identifiers are configured in `scrap.py`.

## Troubleshooting

- **Missing or invalid provider credentials:** Check `LLM_PROVIDER` and the matching API key in `.env`, then restart Streamlit.
- **Ollama model errors:** Verify that Ollama is running and the configured model is available to it.
- **Request too large or token limit errors:** Try a narrower extraction request or reduce the page limit. Large pages can produce large model requests.
- **No records or incomplete output:** Confirm the URL loads without a browser session, refine the requested fields, and review the run logs. Some sites render content with JavaScript, which a simple HTTP request may not include.
- **Invalid JSON:** Retry the extraction or ask for fewer fields. The app attempts to repair a response that is not valid JSON.

## Project files

```text
.
├── app.py                  # Streamlit interface
├── scrap.py                # Scraping tool and model-driven extraction flow
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Pip dependencies
└── src/extract_data/       # Python package
```

## Responsible use

Use this application only on websites you are permitted to access. Follow the target site's terms, robots policy, and applicable rate limits. Scraped page content is treated as untrusted data by the extraction instructions.
