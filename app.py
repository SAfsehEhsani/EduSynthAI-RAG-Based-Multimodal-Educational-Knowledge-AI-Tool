import streamlit as st
from rag_core import process_rag_query # Import the core logic function
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="World Knowledge RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aesthetic UI Functions ---

def set_styles():
    """Applies custom CSS for aesthetic look and feel."""
    st.markdown("""
        <style>
        .main {
            padding: 20px;
            background-color: var(--background-color); /* Use Streamlit theme variable */
            color: var(--text-color); /* Use Streamlit theme variable */
        }
        .stTextArea label, .stSelectbox label, .stButton button {
            font-weight: bold;
        }
        .stTextArea textarea {
             min-height: 150px; /* Make text area larger */
             border-radius: 8px;
             border: 1px solid var(--border-color); /* Use Streamlit theme variable */
             padding: 10px;
             background-color: var(--secondary-background-color); /* Use Streamlit theme variable */
             color: var(--text-color); /* Use Streamlit theme variable */
        }
        .stButton button {
            background-color: #4CAF50; /* Green */
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
            color: white;
        }
         .stExpander {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px;
            margin-top: 20px;
            background-color: var(--secondary-background-color);
        }
         .stMarkdown {
             word-wrap: break-word; /* Ensure text wraps */
             overflow-wrap: break-word;
         }

         /* Adjustments for Dark/Light mode using Streamlit variables */
        :root {
            --text-color: var(--text-color);
            --background-color: var(--background-color);
            --secondary-background-color: var(--secondary-background-color);
            --border-color: var(--border-color);
        }
        </style>
    """, unsafe_allow_html=True)

# Apply styles
set_styles()


# --- API Key Check ---
# Check if Serper key is available - essential for RAG
serper_key_available = bool(os.getenv("SERPER_API_KEY"))

# Check which LLM keys are available
available_llms = []
if os.getenv("GROQ_API_KEY"):
     available_llms.append("Groq")
if os.getenv("OPENAI_API_KEY"):
     available_llms.append("OpenAI")
if os.getenv("ANTHROPIC_API_KEY"):
     available_llms.append("Anthropic")
if os.getenv("HF_API_KEY"):
     available_llms.append("Hugging Face")

if not available_llms:
     st.error("No LLM API keys found in `.env`. Please add at least one (OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, HF_API_KEY).")
     st.stop()

if not serper_key_available:
    st.warning("Serper API key not found in `.env`. Web search will be unavailable, and the application will rely solely on the LLM's internal knowledge.")


# --- UI Layout ---

st.title("📚 World Knowledge RAG")
st.markdown("""
Explore a vast range of topics (Medicine, Engineering, Law, etc.) powered by Web Search and multiple LLMs.
""")

# LLM Selection in Sidebar or Main Area
# Using sidebar for cleanliness
with st.sidebar:
    st.header("Configuration")
    selected_llm = st.selectbox(
        "Choose your LLM Provider:",
        available_llms
    )
    st.markdown("---")
    st.write("API Keys Needed:")
    st.markdown(f"- Serper API Key: {'✅ Found' if serper_key_available else '❌ Missing'}")
    st.markdown(f"- {selected_llm} API Key: {'✅ Found' if selected_llm in available_llms else '❌ Missing'}")
    st.markdown("---")
    st.info("Note: Costs may apply depending on your API plan and usage.")


# Main content area
user_query = st.text_area("Enter your question:", height=150, key="user_query_input")

if st.button("Get Answer"):
    if not user_query:
        st.warning("Please enter a question.")
    elif selected_llm not in available_llms:
         st.warning(f"API key for {selected_llm} is missing in `.env`.")
    else:
        # Add a spinner while processing
        with st.spinner("Searching and generating response..."):
            response, sources = process_rag_query(user_query, selected_llm)

        st.markdown("---")
        st.subheader("Generated Answer:")
        st.markdown(response) # Use markdown to render potential formatting

        if sources:
            st.subheader("Sources:")
            # Use an expander for sources to keep the output clean initially
            with st.expander("Click to see sources"):
                for i, source in enumerate(sources):
                     # Check if it's a valid URL format before making it a link
                     if source and source.startswith("http"):
                          st.markdown(f"{i+1}. [{source}]({source})")
                     else:
                          st.write(f"{i+1}. {source}")
        elif serper_key_available:
             st.info("No specific sources found for the relevant chunks.")
        else:
             st.info("Sources are not available because the Serper API key is missing, and web search was not performed.")

# Add some footer or aesthetic elements
st.markdown("""
    <br><br>
    <div style='text-align: center; opacity: 0.7;'>
        <p>Powered by Serper, various LLMs, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Streamlit handles dark/light mode automatically based on user's browser/system settings
# You can also configure themes in .streamlit/config.toml if needed for more control