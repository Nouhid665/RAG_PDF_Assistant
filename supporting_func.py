from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:3b"


#makes embeddings + database
def create_vector_store(file_path):
    #load pdf 
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    #Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split = text_splitter.split_documents(docs)

    #creating embeddings 
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)# creating the instance of the embidding model

    #database 
    vector_store = Chroma.from_documents(documents=split,
        embedding=embedding,
        collection_name="rag_collection",      # ADD THIS
        persist_directory="./chroma_db")#make embedding in splits 
    return vector_store


#builds RAG pipeline
def create_rag_chain(vector_store):
    llm = OllamaLLM(model=LLM_MODEL)

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Your goal is to provide a detailed and comprehensive answer.
    Extract all relevant information from the context to formulate your response.
    Think step by step and structure your answer logically.
    If the context does not contain the answer to the question, state that the information is not available in the provided context.

    <context>
    {context}
    </context>

    Question: {input}
    """)
    # retriever
    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    #passing documents to LLM 
    document_chain = create_stuff_documents_chain(llm, prompt)#Take the retrieved chunks and stuff them into the prompt context

    #final step
    retrival_chain = create_retrieval_chain(retriever, document_chain)

    return retrival_chain
