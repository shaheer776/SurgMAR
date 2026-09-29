from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
import os
from bs4 import BeautifulSoup
import requests
from langchain_core.prompts import PromptTemplate
import warnings

# Suppress LangChainDeprecationWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")

OPENAI_API_KEY = ''

CUSTOM_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a trusted medical and surgical Retrieval-Augmented Generation expert. Below is some context from your knowledge base, followed by a multiple-choice question.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    1. Do NOT repeat or restate the ANY part question or multiple-choice answer in your answer.
    2. If the questions require the image to be analyzed to determine the answer (e.g. questions on surgical phase or surgical step), respond with "No relevant data found."
    3. Do NOT provide the letter of the correct answer. Just return relevant information and the context.
    4. If the context does not allow you to determine an answer, respond with "No relevant data found."

    Answer:
    """,
)

def build_qa_chain(retriever, openai_api_key):
    """
    Builds a RetrievalQA chain using a custom prompt that ensures
    the final answer does not merely repeat the question.
    """
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        temperature=0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,  # or True if you still want them, but not printed
        chain_type_kwargs={
            "prompt": CUSTOM_QA_PROMPT
        }
    )
    return qa_chain

# URLs to fetch knowledge from (This list is non-exhaustive. Feel free to add more links.)
# The links are just some example html pages which we used
URL_LIST = [
    "https://medlineplus.gov/ency/article/000380.htm",  # Prostate-related
    "https://seer.cancer.gov/statfacts/html/lungb.html",  # Lung cancer facts
    "https://seer.cancer.gov/statfacts/html/prost.html"
]

def fetch_raw_text(url):
    """
    Fetches and extracts raw text from a given URL.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            return text
        else:
            print(f"[ERROR] Failed to retrieve {url}")
            return None
    except Exception as e:
        print(f"[ERROR] Error fetching {url}: {e}")
        return None
def query_rag(query):
    """
    Queries each URL separately and extracts unique, source-specific answers.
    """
    results = {}

    for url in URL_LIST:
        raw_text = fetch_raw_text(url)

        if raw_text is None:
            results[url] = "No relevant data found."
            continue

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_docs = text_splitter.split_text(raw_text)

        # Convert to LangChain Document format
        documents = [Document(page_content=chunk, metadata={"source": url}) for chunk in split_docs]

        # Embedding and FAISS vector storage (URL-specific)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.50}
        )

        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"Retrieved {retrieved_docs} ")

        if not retrieved_docs:  # No relevant documents retrieved
            results[url] = "No relevant data found."
            continue

        # Query each document separately
        qa_chain = build_qa_chain(retriever, OPENAI_API_KEY)

        # Retrieve answer
        result = qa_chain.invoke({"query": query})  # Use `.invoke()` instead of `__call__`
        results[url] = result["result"]

    # Format and return results
    formatted_results = "\n\n".join([f"{url}:\n{answer}" for url, answer in results.items()])
    return formatted_results
