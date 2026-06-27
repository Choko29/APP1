# rag_chain.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import DB_PATH, EMBEDDING_MODEL, LLM_MODEL, SEARCH_RESULTS_COUNT

def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

def format_documents(documents):
    return "\n\n".join([doc.page_content for doc in documents])

def format_chat_history(chat_history):
    if not chat_history:
        return "წინა დიალოგი ჯერ არ არსებობს."
    formatted_turns = []
    for turn in chat_history[-3:]:
        formatted_turns.append(f"მომხმარებელი: {turn['question']}\nასისტენტი: {turn['answer']}")
    return "\n".join(formatted_turns)

def answer_question(question, chat_history):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=SEARCH_RESULTS_COUNT)
    context = format_documents(docs)
    formatted_history = format_chat_history(chat_history)
    
    template = """შენ ხარ AI ჩატბოტი, რომელიც პასუხობს მომხმარებლის შეკითხვებს მოცემული ტექსტური დოკუმენტის მიხედვით.
    წესები:
      - მთავარი პასუხი უნდა დაეყრდნოს დოკუმენტიდან მოძიებულ კონტექსტს.
      - conversation history გამოიყენე მხოლოდ იმისთვის, რომ გაიგო რას გულისხმობს მომხმარებელი.
      - თუ მომხმარებელი ამბობს "ეს", "ის", "წინა", "მეტი ამაზე", გამოიყენე წინა დიალოგი.
      - თუ პასუხი დოკუმენტში არ არის, დაწერე: "ამ დოკუმენტში ამ კითხვაზე პასუხი არ არის მოცემული."
      - არ დაამატო ისეთი ფაქტები, რომლებიც დოკუმენტში არ წერია.
      - უპასუხე ქართულად.

    წინა დიალოგი: {chat_history}
    დოკუმენტიდან მოძიებული კონტექსტი: {context}
    მომხმარებლის ახალი შეკითხვა: {question}
    პასუხი: """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | ChatOllama(model=LLM_MODEL, temperature=0.2) | StrOutputParser()
    return chain.invoke({"context": context, "chat_history": formatted_history, "question": question})
