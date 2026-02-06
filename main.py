from pathlib import Path
import re
import pdfplumber
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    text= word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]

    text = ' '.join(text)
    return text

def upload_documents(uploaded_file):
    file_path = docs_store / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path




def search(querry,documents):
    cleaned_querry = preprocess_text(querry)
    querry_vector = model.encode(cleaned_querry)

    querry_vector = querry_vector.reshape(1, -1)

    results= []

    for document in documents:

        document_vector = document['embedding'].reshape(1,-1)

        similarity = cosine_similarity(querry_vector, document_vector)[0][0]

        results.append({"name" : document["name"] , "similarity" : similarity})
    

    results = sorted(results, key= lambda x : x["similarity"], reverse = True )


    for result in results: 
        print(f"Document: {result['name']}")
        print(f"Similarity: {result['similarity']*100:.2f}")
        print("-"*30)

p = Path('.')
paths = []
model = SentenceTransformer('all-MiniLM-L6-v2')

for file in p.iterdir():
    if (file.is_file() and (file.suffix =='.pdf' or file.suffix == '.txt')):
        paths.append(file.absolute())
        print(file.name)

documents= []

for path in paths:
    if path.suffix == '.txt':
        with open(path, 'r', encoding = 'utf-8' ) as f:
            content = f.read()
            cleaned_content = preprocess_text(content)
            encoding = model.encode(cleaned_content)
            documents.append({"name": path.name, "content": cleaned_content, "embedding" : encoding})
    if path.suffix == '.pdf':
        with pdfplumber.open(path) as pdf:
            content = ""
            for page in pdf.pages:
                content += page.extract_text() + "\n"
            cleaned_content =  preprocess_text(content)
            encoding =model.encode(cleaned_content)
            documents.append({"name": path.name, "content": cleaned_content, "embedding" : encoding})


print(documents[0])

search("unity game development physics", documents)