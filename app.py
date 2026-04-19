import streamlit as st

from src.assistant import answer_question
from src.config import AppConfig, LLMConfig, get_available_llm_choices
from src.schemas import AssistantResponse


CLIENT_OPTIONS = {
    "All clients": None,
    "Fjordmat": "fjordmat",
    "Spareklar": "spareklar",
    "Nordvik": "nordvik",
    "Skytjenester": "skytjenester",
}

STATUS_LABELS = {
    "answered": "Answered",
    "retrieval_only": "Retrieval only",
    "no_results": "No results",
    "configuration_error": "Configuration error",
    "runtime_error": "Runtime error",
}


def render_sources(sources: list[str]) -> None:
    if not sources:
        st.caption("No sources.")
        return

    for source in sources:
        st.markdown(f"- `{source}`")


def render_trace(response: AssistantResponse) -> None:
    retrieved_context = response.retrieved_context

    if response.model_name:
        st.text(f"Model: {response.model_name}")

    st.text(f"Status: {STATUS_LABELS.get(response.status, response.status)}")

    if response.error:
        st.text(f"Error: {response.error}")

    if not retrieved_context:
        return

    client_filter = retrieved_context.client_filter_used or "None"
    st.text(f"Client filter: {client_filter}")

    for note in retrieved_context.retrieval_notes:
        st.text(note)

    if retrieved_context.context_block:
        st.text_area(
            "Retrieved context",
            retrieved_context.context_block,
            height=320,
            disabled=True,
        )


def render_prompt(prompt: str | None) -> None:
    if not prompt:
        st.caption("No prompt was built.")
        return

    st.text_area("Prompt", prompt, height=320, disabled=True)


def render_response(response: AssistantResponse) -> None:
    status_label = STATUS_LABELS.get(response.status, response.status)

    if response.status == "answered":
        st.success(status_label)
    elif response.status == "retrieval_only":
        st.info(status_label)
    elif response.status == "no_results":
        st.warning(status_label)
    elif response.status == "configuration_error":
        st.error(status_label)
    else:
        st.error(status_label)

    st.markdown(response.answer)

    with st.expander("Sources", expanded=bool(response.sources)):
        render_sources(response.sources)

    with st.expander("Retrieval trace"):
        render_trace(response)

    with st.expander("Prompt"):
        render_prompt(response.prompt)


def main() -> None:
    st.set_page_config(page_title="RAG Knowledge Assistant", page_icon=":material/search:")
    st.title("RAG Knowledge Assistant")

    with st.sidebar:
        selected_client_label = st.selectbox("Client", list(CLIENT_OPTIONS.keys()))
        selected_client = CLIENT_OPTIONS[selected_client_label]
        llm_choices = get_available_llm_choices()
        llm_choice_labels = ["Retrieval only"] + [choice.label for choice in llm_choices]
        selected_llm_label = st.selectbox("LLM provider", llm_choice_labels)
        selected_llm_config = None

        if selected_llm_label != "Retrieval only":
            selected_llm_config = next(
                choice.config for choice in llm_choices if choice.label == selected_llm_label
            )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "response" in message:
                render_response(message["response"])
            else:
                st.markdown(message["content"])

    question = st.chat_input("Ask about client documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching client knowledge base..."):
            app_config = AppConfig(
                llm=selected_llm_config
                if selected_llm_config
                else LLMConfig(
                    provider="openai_compatible",
                    model_name="your-llm-model",
                    base_url="",
                    api_key="",
                )
            )
            response = answer_question(
                question,
                client_filter=selected_client,
                app_config=app_config,
            )
        render_response(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "response": response,
        }
    )


if __name__ == "__main__":
    main()
