
import getpass
import os
# Load the .env file need to add following two lines
from dotenv import load_dotenv
load_dotenv()
####################
from langchain_groq import ChatGroq
import requests
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        dl_meta  = doc.metadata.get("dl_meta", {})
        parts.append(f"[{i}]: \n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def rag(query: str) -> str:
    docs = retriever.invoke(query)
    context = format_docs(docs)
    prompt_value = RAG_PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(prompt_value)
    return response.content


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Hi This is the llm PyCharm')
    resp = llm.invoke("What is the capital of India?")
    print(f"Response Content is {resp.content}")



    #Create document
    WEB_URL = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/atliqai_hr_policies.txt"

    response = requests.get(WEB_URL)
    response.raise_for_status()

    docs = [Document(page_content=response.text, metadata={"source": WEB_URL})]
    print(f"Page Metadata:\n {docs[0].metadata}")
    print(f"Page Content: \n{docs[0].page_content}")


    # Text Splitter
    from langchain_text_splitters import CharacterTextSplitter

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
    )
    documents = text_splitter.split_documents(docs)
    len(documents)
    print(f"Split {len(documents)} documents")

    for document in documents:
        print(f"Content of doc: {document.page_content}")
        print(f"Meta Data  of doc: {document.metadata}")
        break

    # Test format_docs function
    # print(f"Formated Document:\n {format_docs(documents)}")


    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore

    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    COLLECTION_NAME = "hr_docs"
    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedder,
        path="/tmp/my_lang_vs",
        collection_name=COLLECTION_NAME
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the context below. Cite section names. Say 'I don't know' if unsure."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    for q in [
        "How many casual leaves am I entitled to?",
        "What is the notice period for Band 4 employees?",
        "How long is the probation period?",
    ]:
        print(f"\nQ: {q}\nA: {rag(q)}\n")

    vectorstore.client.close()
