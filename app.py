import streamlit as st

from rag.generation import generate_answer


st.set_page_config(
    page_title="Suzuki RAG Chatbot",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Suzuki Chatbot")
st.caption("RAG-powered Suzuki automotive knowledge assistant")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], 1):
                    metadata = source.get("metadata", {})
                    source_name = metadata.get("source", "Unknown source")
                    url = metadata.get("url", "")
                    entity = metadata.get("target_entity", "")

                    st.markdown(f"**{i}. {source_name}**")

                    if entity:
                        st.caption(f"Entity: {entity}")

                    if url:
                        st.markdown(f"[Open source]({url})")


# User input
question = st.chat_input("Ask about a Suzuki vehicle...")


if question:
    # Show user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching Suzuki knowledge..."):
            try:
                answer, sources = generate_answer(question)

                st.markdown(answer)

                with st.expander("Sources"):
                    for i, source in enumerate(sources, 1):
                        metadata = source.get("metadata", {})
                        source_name = metadata.get("source", "Unknown source")
                        url = metadata.get("url", "")
                        entity = metadata.get("target_entity", "")

                        st.markdown(f"**{i}. {source_name}**")

                        if entity:
                            st.caption(f"Entity: {entity}")

                        if url:
                            st.markdown(f"[Open source]({url})")

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                error_message = f"Error: {type(e).__name__}: {e}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })