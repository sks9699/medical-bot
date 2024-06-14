from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import Pinecone
app = Flask(__name__)




# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pc = Pinecone(api_key="8d8905b7-a0c9-4765-b687-639f837aa7dd")
index = pc.Index("medical-bot")

#Loading the index
# docsearch=Pinecone.from_existing_index(index_name, embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

from langchain.vectorstores import Pinecone as LangchainPinecone
text_field = "text"
vectorstore = LangchainPinecone(
    index=index,
    embedding_function=embeddings.embed_query,
    text_key=text_field
)



qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever(top_k=3),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(dir(qa))
    result=qa.invoke({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)