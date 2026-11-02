import json

import streamlit as st

from src.extract_data.scrap import scrape_agent

st.title("Scrap Agent")
st.write("Extract information from a webpage and choose how to display it.")

with st.form("extraction_form"):
    url = st.text_input("Enter URL", placeholder="https://example.com/page")
    question = st.text_input(
        "What do you want to extract?",
        placeholder="For example: Extract each pet's name, age, and species",
    )
    output_format = st.selectbox("Select output format", ["text", "json", "table"])
    show_page_text = st.checkbox("Show page text sent to the agent in logs", value=False)
    page_limit = st.slider("Page limit", min_value=1, max_value=100, value=10)
    submitted = st.form_submit_button("Extract")

if submitted:
    if not url.strip():
        st.error("Enter a URL to scrape.")
    elif not question.strip():
        st.error("Describe what you want to extract.")
    else:
        st.subheader("Run logs")
        log_lines = []
        log_view = st.empty()

        def show_log(message: str) -> None:
            log_lines.append(message)
            log_view.code("\n\n".join(log_lines), language="text")

        with st.spinner("Reading the page and extracting information..."):
            try:
                result = scrape_agent(
                    question=question,
                    url=url.strip(),
                    output_format=output_format,
                    log_callback=show_log,
                    include_page_text_in_logs=show_page_text,
                    page_limit=page_limit
                )
                st.subheader("Result")
                if output_format == "json":
                    st.json(json.loads(result))
                elif output_format == "table":
                    st.markdown(result)
                else:
                    st.write(result)
            except json.JSONDecodeError:
                st.error("The model returned invalid JSON. Please try the request again.")
            except Exception as error:
                st.error(f"Could not extract data: {error}")
