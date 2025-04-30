import os
import chromadb
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import WebBaseLoader
import openai
import anthropic
from groq import Groq
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# --- API Clients (Instantiate) ---
# Initialize clients only if keys are available
openai_client = None
if os.getenv("OPENAI_API_KEY"):
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

anthropic_client = None
if os.getenv("ANTHROPIC_API_KEY"):
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

groq_client = None
if os.getenv("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

hf_client = None
if os.getenv("HF_API_KEY"):
     hf_client = InferenceClient(token=os.getenv("HF_API_KEY")) # Requires HF_API_KEY for non-rate-limited or private models

# Serper API Key
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Embedding Model (using a local SentenceTransformer model)
# You can choose a different model depending on performance/accuracy needs
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
try:
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
except Exception as e:
     print(f"Error loading embedding model {EMBEDDING_MODEL_NAME}: {e}")
     embedding_model = None # Handle cases where model download/load fails


# --- RAG Functions ---

def get_search_results(query: str, num_results: int = 5):
    """Uses Serper API to get web search results (URLs and snippets)."""
    if not SERPER_API_KEY:
        print("Serper API key not found.")
        return []

    search_url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(search_url, headers=headers, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        urls = []
        if data.get("organic"):
            for result in data["organic"][:num_results]:
                urls.append(result.get("link"))
        # Add results from answerBox if available
        if data.get("answerBox") and data["answerBox"].get("snippet"):
             urls.insert(0, data["answerBox"].get("link") or "N/A") # Prioritize answer box source if exists

        return [url for url in urls if url] # Filter out None values

    except requests.exceptions.RequestException as e:
        print(f"Error calling Serper API: {e}")
        return []

def fetch_and_chunk_documents(urls: list):
    """Fetches content from URLs and splits it into chunks."""
    if not urls:
        return [], []

    all_chunks = []
    sources = []

    # Use Langchain's WebBaseLoader for fetching and parsing
    loader = WebBaseLoader(urls)
    try:
        docs = loader.load()
    except Exception as e:
        print(f"Error loading documents from URLs: {e}")
        return [], []

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Store original source for each chunk (assuming doc.metadata['source'] exists)
    for chunk in chunks:
        all_chunks.append(chunk.page_content)
        sources.append(chunk.metadata.get('source', 'Unknown Source'))

    return all_chunks, sources # Return chunk text and corresponding sources

def create_and_query_vector_db(chunks: list, query: str):
    """Creates an in-memory Chroma DB and queries it."""
    if not chunks or embedding_model is None:
        return [], []

    # Create a temporary, in-memory Chroma collection
    # We use a simple name; for persistence, you'd configure a directory
    collection_name = "web_knowledge_collection"
    # Using ephemeral client means data is lost after the program stops
    client = chromadb.EphemeralClient()
    # Get or create the collection; specifying embedding function
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_model # Pass the embedding function
    )

    # Add chunks to the collection
    # ChromaDB requires unique IDs, text content, and optionally metadata/embeddings
    # Since we pass an embedding function, ChromaDB will compute embeddings
    try:
         collection.add(
             documents=chunks,
             ids=[f"doc_{i}" for i in range(len(chunks))] # Simple unique IDs
         )
    except Exception as e:
         print(f"Error adding documents to ChromaDB: {e}")
         # Attempt to clear the collection and retry or just return empty
         try:
              client.delete_collection(collection_name)
              print(f"Cleared collection {collection_name} due to error.")
         except:
              pass # Ignore if deletion also fails
         return [], []


    # Query the collection
    try:
        results = collection.query(
            query_texts=[query],
            n_results=5 # Get top 5 relevant chunks
        )
        if results and results.get("documents") and results["documents"][0]:
            # results['documents'] is a list of lists (one inner list per query)
            relevant_chunks = results["documents"][0]
            # We'd ideally also get the sources here if they were stored in metadata
            # For simplicity, we'll just return the chunk text
            return relevant_chunks, [] # Return relevant text chunks
        else:
             return [], []

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return [], []


def build_prompt(query: str, relevant_docs: list):
    """Builds a prompt for the LLM incorporating the query and relevant documents."""
    context = "\n\n".join(relevant_docs)

    if not context:
         # Fallback prompt if no relevant documents are found
         return f"Based on your internal knowledge, please answer the following question: {query}"


    prompt = f"""You are an AI assistant that provides information based on the context provided.
    Answer the user's question comprehensively and accurately using ONLY the provided context.
    If the answer cannot be found in the context, state that you cannot answer based on the provided information.

    Context:
    {context}

    Question: {query}

    Answer:"""
    return prompt

def get_llm_response(prompt: str, llm_provider: str, api_keys: dict):
    """Gets a response from the selected LLM provider."""

    if llm_provider == "OpenAI" and openai_client:
        try:
            chat_completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo", # Or another available model like gpt-4
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI API: {e}"

    elif llm_provider == "Anthropic" and anthropic_client:
         try:
             message = anthropic_client.messages.create(
                 model="claude-3-sonnet-20240229", # Or another available model
                 max_tokens=1000,
                 messages=[
                     {"role": "user", "content": prompt}
                 ]
             )
             return message.content[0].text if message.content else "No response from Anthropic."
         except Exception as e:
             return f"Error calling Anthropic API: {e}"

    elif llm_provider == "Groq" and groq_client:
         try:
             chat_completion = groq_client.chat.completions.create(
                 messages=[
                     {"role": "user", "content": prompt}
                 ],
                 model="llama3-8b-8192", # Or another available model like mixtral-8x7b-32768
             )
             return chat_completion.choices[0].message.content
         except Exception as e:
             return f"Error calling Groq API: {e}"

    elif llm_provider == "Hugging Face" and hf_client:
         # This uses a general inference endpoint. You might specify a model ID.
         # Free endpoints are rate-limited.
         try:
              # Using a basic text generation task
              # The prompt needs to be suitable for the underlying HF model
              # This is a simplified example, prompt format varies by model
              response = hf_client.text_generation(
                  prompt=prompt,
                  max_new_tokens=1000
                  # add other parameters like temperature, etc.
              )
              return response
         except Exception as e:
             return f"Error calling Hugging Face Inference API: {e}. Ensure HF_API_KEY is set and a suitable model/task is targeted."

    else:
        return f"Selected LLM provider '{llm_provider}' is not available or API key is missing."


def process_rag_query(query: str, llm_provider: str):
    """Main function to orchestrate the RAG process."""
    if not query:
        return "Please enter a query."

    # 1. Web Search
    print(f"Searching the web for: {query}")
    urls = get_search_results(query)
    if not urls:
        # Fallback to LLM's internal knowledge if search fails
        print("Web search failed or returned no URLs. Using LLM internal knowledge.")
        # Build prompt without context
        fallback_prompt = build_prompt(query, []) # Passes empty context
        response = get_llm_response(fallback_prompt, llm_provider, {}) # Pass empty keys as they are loaded globally
        return response, []


    print(f"Found URLs: {urls}")

    # 2. Fetch and Chunk Documents
    print("Fetching and chunking documents...")
    chunks_text, sources = fetch_and_chunk_documents(urls)
    if not chunks_text:
        # Fallback if document fetching/chunking fails
        print("Failed to fetch or chunk documents. Using LLM internal knowledge.")
        fallback_prompt = build_prompt(query, []) # Passes empty context
        response = get_llm_response(fallback_prompt, llm_provider, {})
        return response, []

    print(f"Created {len(chunks_text)} chunks.")

    # 3. Create Vector DB and Query
    print("Creating and querying vector database...")
    # Note: We don't need query embedding explicitly here; ChromaDB handles it
    # using the embedding_model function passed during collection creation.
    # The query function automatically embeds the query_texts.
    relevant_chunks_text, _ = create_and_query_vector_db(chunks_text, query)

    if not relevant_chunks_text:
        # Fallback if vector search finds no relevant chunks
        print("Vector database search returned no relevant chunks. Using LLM internal knowledge.")
        fallback_prompt = build_prompt(query, []) # Passes empty context
        response = get_llm_response(fallback_prompt, llm_provider, {})
        return response, []

    print(f"Found {len(relevant_chunks_text)} relevant chunks.")

    # 4. Build Prompt
    print("Building LLM prompt...")
    prompt = build_prompt(query, relevant_chunks_text)
    # print("--- Generated Prompt ---")
    # print(prompt)
    # print("------------------------")


    # 5. Get LLM Response
    print(f"Getting response from {llm_provider}...")
    response = get_llm_response(prompt, llm_provider, {}) # Pass empty keys, loaded globally

    # Return the response and the list of sources (URLs fetched)
    # Note: Mapping relevant chunks back to specific source URLs requires more complex metadata handling.
    # For simplicity, we return the initial list of fetched URLs.
    unique_sources = list(set(sources)) # Get unique sources from *all* chunks (a simplification)
    return response, unique_sources

# Example of how to use this function (can be removed/commented for Streamlit)
# if __name__ == "__main__":
#     test_query = "Explain the concept of black holes in astrophysics."
#     test_provider = "Groq" # Change to "OpenAI", "Anthropic", etc.

#     print(f"Running RAG query for: {test_query} using {test_provider}")
#     response, sources = process_rag_query(test_query, test_provider)

#     print("\n--- Response ---")
#     print(response)

#     print("\n--- Sources ---")
#     if sources:
#         for i, source in enumerate(sources):
#             print(f"{i+1}. {source}")
#     else:
#         print("No specific sources identified.")