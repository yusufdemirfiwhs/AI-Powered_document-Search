
import streamlit as st
from pathlib import Path
import pdfplumber
import re
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')


st.set_page_config(page_title="AI Search", page_icon="🐢")

@st.cache_resource 


def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    text= word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]

    text = ' '.join(text)
    return text


def split_text_into_chunks(text, chunk_size=500, overlap=50):
    
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def read_documents():
    p = Path('.')
    documents = []
    
    
    files = list(p.glob('*.pdf')) + list(p.glob('*.txt'))
    
    if not files:
        st.warning("There are no PDF or TXT")
        return []

    progress_text = "Document processing in progress..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, file in enumerate(files):
        content = ""
        if file.suffix == '.txt':
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
        elif file.suffix == '.pdf':
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        content += extracted + " "
        
        
        if content:
            cleaned_content = re.sub(r'\s+', ' ', content).strip()
            cleaned_content = cleaned_content.lower()
            chunks = split_text_into_chunks(cleaned_content, chunk_size=500, overlap= 50)

            for chunk in chunks:
                embedding = model.encode(chunk)
                documents.append({
                    "name": file.name,
                    "content" : chunk,
                    "embedding": embedding,
                    "path" : str(file.resolve())
                })
          
    
        my_bar.progress((i + 1) / len(files), text=progress_text)
    
    my_bar.empty() 
    return documents


st.title("AI Document Search")
st.write("Search through your PDF and TXT documents using AI-powered semantic search.")


if 'docs' not in st.session_state:
    with st.spinner('Documents are being indexed...'):
        st.session_state['docs'] = read_documents()
    
    if(st.session_state['docs']):
        unique_files = {doc['name'] for doc in st.session_state['docs']}
    st.success(f"{len(unique_files)} documents indexed!")

query = st.text_input("What would you like to search for ?", placeholder="Enter your search querry here...")

if st.button("Search") or query:
    if query:
        
        cleaned_query = preprocess_text(query)
        query_vector = model.encode(cleaned_query).reshape(1, -1)
        
        results = []
        for doc in st.session_state['docs']:
            doc_vector = doc['embedding'].reshape(1, -1)
            score = cosine_similarity(query_vector, doc_vector)[0][0]
            results.append({"name": doc['name'], "score": score, "content": doc['content']})
        
        
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        seenfiles = set()
        unique_results = []

        for res in results:
            if( res['name'] not in seenfiles):
                unique_results.append(res)
                seenfiles.add(res["name"])
        
        st.subheader("Search results")
        found = False
        for res in unique_results:
            
            if res['score'] > 0.3:
                found =True 
                with st.expander(f"📄 {res['name']} - Score: %{res['score']*100:.2f}"):
                    st.markdown(f"**Content Summary:**")
                    st.write(res['content'][:500] + "...")
                    
                    try:
                        file_path = res['name']
                        with open(file_path, 'rb') as f:
                            btn = st.download_button(
                                label = "Download Document",
                                data  = f,
                                file_name = res['name'],
                                mime = "application/pdf" if res['name'].endswith('.pdf') else "text/plain"
                            )
                    except Exception as e:
                        st.error(f"Error Downloading file {e}")
            else:
                
                pass
    else:
        st.warning("Plese enter a search querry.")


with st.sidebar:
    st.header("About")
    st.write("This project is built using Python, Sentence-Transformers, and Streamlit.")
    if st.button("Refresh Documents"):
        st.session_state.pop('docs', None)
        st.rerun()