import os # Gives the program access to the windows file system- need it to check if index exists and any environment variables
from dotenv import load_dotenv # Gives the program the ability to read the .env file to check for env variables

import streamlit as st # Using streamlit to build a simple web app and st becomes the alias to call

# These are the LlamaIndex pieces
from llama_index.llms.openai import OpenAI # Gives us access to the openai llms
from llama_index.embeddings.openai import OpenAIEmbedding # Gives us access to embedding models to turn the text into vectors
from llama_index.readers.wikipedia import WikipediaReader # Gives us access to the wikipedia reader which downloads wikipedia articles
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage # The searchable index of embedded chunks, with the ability to save and reload the index on disk

load_dotenv() # Load the .env file into the environment to be called on

INDEX_DIR = 'wiki_rag' # The folder in our project where our vector store will be held
PAGES = ["Chicago Bears", "National Football League", "National Football Conference", "NFL playoffs"] # List of wikipedia pages we want our reader to look at

# Streamlit would usually rerun our script every time every time the user interacts with the page, so we're caching the indexing to not have to rebuild it every click.
@st.cache_resource
def get_index():
    # Check if a previous run of the program already build the index. If yes, then just load the saved index and return
    if os.path.isdir(INDEX_DIR):
        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage)

    docs = WikipediaReader().load_data(pages=PAGES, auto_suggest = False) # Download the 4 articles, and only using the 4 we requested, no finding alternates.
    embedding_model = OpenAIEmbedding(model = "text-embedding-3-small") # Creating our embedding model instance
    index = VectorStoreIndex.from_documents(docs, embed_model = embedding_model) # Create the index by splitting the articles into chunks, embed each chunk via the embedding model, and store the chunks and vectors into the index. 
    index.storage_context.persist(persist_dir = INDEX_DIR) # Saves the index to the directory, so the next run skips the recreation

    return index

@st.cache_resource
def get_query_engine():
    index = get_index() # Get the cached index. 

    llm = OpenAI(model = 'gpt-5.6-luna', temperature = 0) # Create the llm instance

    return index.as_query_engine(llm = llm, similarity_top_k = 3) # Wrap the llm and index into a RAG pipeline. Only retrieve the 3 chunks most similar to the prompt.

# Display the contents to streamlit
def main():
    st.title('Wikipedia RAG Application')
    question = st.text_input('Ask a question')

    if st.button('Submit') and question: # Check if the user entered text and pressed submit
        # Run the rag pipeline to do a similar search between the user's prompt and the index. The chunks + our prompt are then sent to the model. 
        with st.spinner('Thinking...'):
            qa = get_query_engine()
            response = qa.query(question)

        st.subheader('Answer')
        st.write(response.response) # Writing the model's response as a string to the ui
        st.subheader('Retrieved context')
        for src in response.source_nodes: # Retrieve the embedding chunks found 
            st.markdown(src.node.get_content()) # Display the chunk's text

if __name__ == '__main__':
    main()