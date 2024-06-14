from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)




from pinecone import Pinecone

# PINECONE_API_KEY = "8d8905b7-a0c9-4765-b687-639f837aa7dd"
# # PINECONE_API_ENV = "gcp-starter"
pc = Pinecone(api_key="8d8905b7-a0c9-4765-b687-639f837aa7dd")
index = pc.Index("medical-bot")




extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key="8d8905b7-a0c9-4765-b687-639f837aa7dd")
index = pc.Index("medical-bot")



text = [t.page_content for t in text_chunks]
metadata_list = [{"text": chunk.page_content} for chunk in text_chunks]
## cretaing vector
embeddings_list = [model.embed_query(text) for text in text]

#Creating Embeddings for Each of The Text Chunks & storing
# docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
index_name="medical-bot"

# Create vectors
vectors = []
for i, emb in enumerate(embeddings_list):
    vector = {
        "id": f"vec{i}",
        "values": emb , # Convert embedding tuple to list
        "metadata": metadata_list[i]
    }
    vectors.append(vector)
batch_size=100
# Upsert vectors in batches
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.upsert(batch, index_name=index_name, namespace="ns1")