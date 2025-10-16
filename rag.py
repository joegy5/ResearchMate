import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment.\n"
    )

PDF_PATH = r"paper1.pdf"

LLM_MODEL_NAME = "llama-3.3-70b-versatile"

PERSIST_DIRECTORY = "./chroma_db"
RECREATE_VECTOR_DB = False

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return "\n\n".join(pages)

def build_vectorstore_from_pdf(pdf_path, persist_dir=PERSIST_DIRECTORY,
                               chunk_size=1000, chunk_overlap=200,
                               force_recreate=False):
    print("Extracting and embedding text from paper...")
    text = extract_text_from_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if force_recreate and os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

def make_chat_chain(vectordb, llm_model=LLM_MODEL_NAME):
    llm = ChatGroq(model=llm_model, temperature=0.0)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":4})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    return chain

def interactive_chat(chain):
    print("\nType your questions about the paper below.")
    print("Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in ["exit", "quit"]:
            print("\nChat ended.")
            break
        res = chain.run({"question": q})
        answer = res if isinstance(res, str) else res.get("answer", str(res))
        print("\nBot:", answer, "\n")

if __name__ == "__main__":
    db = build_vectorstore_from_pdf(PDF_PATH, force_recreate=RECREATE_VECTOR_DB)
    chain = make_chat_chain(db)
    interactive_chat(chain)
