from flask import Flask, request, jsonify
import os
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
#from langchain.vectorstores import Chroma
import constants

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = constants.APIKEY
PERSIST = False

# Set up your model, index, chain, etc. outside of the API endpoint to avoid redundancy
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)
chat_history = []

RECOMMENDATION = ""

@app.route('/chatbot/', methods=['POST'])
def chatbot():
    user_input = request.json.get('user_input', '')
    query = user_input + RECOMMENDATION
    result = chain({"question": query, "chat_history": chat_history})
    response = result['answer']
    chat_history.append((query, response))
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5000, debug=True)