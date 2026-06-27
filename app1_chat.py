# app1_chat.py
import os
import streamlit as st
import rag_chain
from config import DB_PATH

st.set_page_config(page_title="RAG Chatbot with Memory", page_icon="💬")
st.title("💬 AI ჩატბოტი მეხსიერებით")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("დაუსვით შეკითხვა ჩატბოტს:")

if st.button("გაგზავნა"):
    if question.strip() and os.path.exists(DB_PATH):
        with st.spinner("ფიქრობს..."):
            answer = rag_chain.answer_question(question, st.session_state.chat_history)
            st.session_state.chat_history.append({"question": question, "answer": answer})
            st.success(answer)
    else:
        st.error("შეიყვანეთ ტექსტი ან შექმენით ბაზა!")

if st.session_state.chat_history:
    with st.expander("💬 დიალოგის ისტორია"):
        for turn in st.session_state.chat_history:
            st.write(f"**სტუდენტი:** {turn['question']}")
            st.write(f"**რობოტი:** {turn['answer']}")
