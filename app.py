import streamlit as st

# app logic
DB_FAISS_PATH = "vectorstores/db_faiss"

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain. llms import CTransformers
from langchain.chains import RetrievalQA

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer. 
Context: {context}
Question: {question}. 
Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])
    return prompt

def load_llm():
    llm = CTransformers(
    model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type = "llama",
    max_new_tokens = 512,
    temperature = 0.1)

    return llm


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents = True,
    chain_type_kwargs = {'prompt': prompt})

    return qa_chain

def qa_bot():
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
db = FAISS. load_local(DB_FAISS_PATH, embeddings)
llm = load_llm()


st.title("💬 Chatbot") 

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    res=final_result(prompt)
    response={"role": "assistant", "content": res['result']}
    st.session_state.messages.append(response)
    st.chat_message("assistant").write(response['content'])