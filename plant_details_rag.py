import json
import os 

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

GEMENI_KEY = os.getenv("GEMINI_API_KEY") # Gets the gemeni api key from the environment variables

# print(GEMENI_KEY)

plant_database_file = "plants.json"

def plant_json_to_text(plant):

    return f"""
    Plant name: {plant['name']} 
    Description: {plant['description']} 
    Scientific name: {plant['scientific_name']}
    Care, light levels: {plant['care']['light']}
    Care, water needs: {plant['care']['water']}
    Care, soil: {plant['care']['soil']}
    Care, prefered temperature and humidity: {plant['care']['temperature_and_humidity']}
    Care, tips: {plant['care']['tips']}
    """

def create_retriever():
    data = json.load(open(plant_database_file)) # reads the json file
    texts = [plant_json_to_text(plant) for plant in data] # reads the text from the json file
    ids = [ plant['id'] for plant in data] # reads the ids from the json file

    documents = [Document(page_content=text, id=id) for text, id in zip(texts, ids)]

    gemeni_embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004"
    ) # creates a gemeni embedding model 

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=gemeni_embeddings,
        persist_directory='./plant_embeddings'
        ) # creates a vector store from the documents
    
    # query = "What plant likes a lot of humidity?"

    # results = vector_store.max_marginal_relevance_search(query=query, k=2) # prints the top two results from the query
    # for result in results:
    #     print(result.page_content)
    # k = get the k best results
    # fetch_k is pick the 15 best from store
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 15}) 
    return retriever


def create_rag_chain(retriever):
    rag_prompt = PromptTemplate(
        template="""Answer the customer's question using the context provided. 
        Reply 'I don't know' if the context can't be used be used to answer the customer's question.
        if the context doesn't direct answer the question, reply with whatever you do know that may be related.


        Use this context to answer the question:
        {context}

        Customer's Question:
        {question}

        Response:
        """,
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.5)


    plant_rag_chain = ( 
        {'context': retriever, 'question': RunnablePassthrough()} # Finds the most valuable documents to be provided as context to the question
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return plant_rag_chain

@tool
def plant_rag(customer_query): # This represents the customers query
    """ Answer general questions about plant care and plant attributes for plants we sell
    And, find plants that match customer requirements, for example, care levels, size, amount of sun/light, water, soil
    """
    return plant_rag_chain.invoke(customer_query)

doc_retriever = create_retriever()
plant_rag_chain = create_rag_chain(doc_retriever)


# customer_query = "What plants like a lot of sun?"
# rag_result = plant_rag_chain.invoke(customer_query)
# print(rag_result)