import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0,
    convert_system_message_to_human=True,
    google_api_key=GOOGLE_API_KEY
)

# Dictionary to manage vectorstores for different websites
vectorstores = {
    "opensenselabs.com": Chroma(persist_directory='chroma_db_index_drupalfit_osl', embedding_function=embedding_model),
    "koshishforindia.org": Chroma(persist_directory='chroma_db_index_drupalfit', embedding_function=embedding_model)
}

# Streamlit app setup
st.set_page_config(page_title="AI-Powered Website Audit Assistant", page_icon="🎨", layout="wide")
st.title("🔧 AI-Powered Website Audit Assistant")
st.write("Interact with an intelligent assistant to gain actionable insights into your website's SEO, performance, and accessibility!")

# Sidebar for user instructions
st.sidebar.header("Instructions")
st.sidebar.write("1. Select the website you want to audit.")
st.sidebar.write("2. Type your query about the website's audit (SEO, performance, accessibility, etc.).")
st.sidebar.write("3. View the assistant's detailed insights and recommendations.")

# Dropdown to select website
selected_website = st.selectbox("Select Website:", options=list(vectorstores.keys()))

# User input
user_query = st.text_input("Ask your question:", placeholder="e.g., How can I improve my website's SEO and reduce TBT?")

if st.button("Get Insights"):
    if user_query.strip():
        try:
            # Use the selected website's vectorstore retriever
            retriever = vectorstores[selected_website].as_retriever(search_kwargs={"k": 5})
            res = retriever.invoke(user_query)
            retrieved_data = str(res)

            # Prepare messages for LLM
            messages = [
                SystemMessage(
                    content=(
                        "You are a website audit assistant. Use only the provided data to answer questions about SEO, performance, and accessibility. "
                        "Do not fabricate or guess answers. If the information is not available in the data, clearly state that you cannot answer. "
                        "Here is the data: " + retrieved_data
                    )
                ),
                HumanMessage(content=user_query)
            ]

            # Generate response using Gemini LLM
            results = llm.invoke(messages).content

            # Display the response
            st.subheader("Assistant's Insights")
            st.write(results)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query.")

# Footer
st.markdown("---")
st.markdown(
    "**Developed by opensenselabs** 🌐 | Powered by Gemini LLM & Chroma VectorStore."
)
