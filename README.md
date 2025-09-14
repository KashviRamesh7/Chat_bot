#Python version 3.10



This Streamlit app is a Document Q&A Chatbot powered by Cohere LLMs and FAISS. It allows users to upload PDF, DOCX, and Excel files, extracts and processes their content, and stores it as embeddings. Using LangChain’s Conversational Retrieval Chain, the bot retrieves relevant text chunks for each query. The chatbot supports multi-turn conversations with preserved context via Streamlit session state. Built with Cohere Embeddings + ChatCohere, it provides fast, intelligent, and document-aware answers.
