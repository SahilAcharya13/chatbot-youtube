import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

# -------------------------------
# Initialize Vector DB in Pinecone
# -------------------------------
def process_transcript_to_pinecone(text, video_id):
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
    
    if not pinecone_api_key or not pinecone_index_name:
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set.")

    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists, else create it
    if pinecone_index_name not in [index.name for index in pc.list_indexes()]:
        print(f"Creating Pinecone index: {pinecone_index_name}...")
        pc.create_index(
            name=pinecone_index_name,
            dimension=384, # all-MiniLM-L6-v2 output dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.create_documents([text])
    
    # Check if this video is already in Pinecone (by namespace) to avoid duplicate indexing
    index = pc.Index(pinecone_index_name)
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})
    
    if video_id not in namespaces.keys():
        print(f"Uploading vectors for video ID: {video_id} to Pinecone namespace.")
        PineconeVectorStore.from_documents(
            docs, 
            index_name=pinecone_index_name, 
            embedding=embeddings, 
            namespace=video_id
        )
    else:
        print(f"Video ID: {video_id} is already indexed in Pinecone.")
        
    return True


# -------------------------------
# Create QA Chain for chatting
# -------------------------------
def create_qa_chain(video_id):
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
    xai_api_key = os.environ.get("XAI_API_KEY")
    
    if not xai_api_key:
        raise ValueError("XAI_API_KEY is not set.")
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(
        index_name=pinecone_index_name, 
        embedding=embeddings, 
        namespace=video_id
    )
    
    # Groq via OpenAI compatible endpoint
    llm = ChatOpenAI(
        api_key=xai_api_key, 
        base_url="https://api.groq.com/openai/v1", 
        model="llama-3.1-8b-instant", 
        max_tokens=1000,
        temperature=0.3
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    system_prompt = (
        "You are an intelligent chatbot assistant. Your primary task is to answer "
        "questions accurately based *only* on the provided YouTube video transcript context. "
        "If you don't know the answer or if the context doesn't have the information, "
        "politely state that you don't know based on the video context. Do not make things up.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return qa_chain

def ask_question(video_id, question):
    qa_chain = create_qa_chain(video_id)
    response = qa_chain.invoke({"input": question})
    return response["answer"]