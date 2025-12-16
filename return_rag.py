from dotenv import load_dotenv
load_dotenv()

import json
import os 

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.tools import tool



GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Gets the gemeni api key from the environment variables
return_policy_document = 'return_policy_document.md'

def create_reciever():

    with open(return_policy_document) as file:
        policy_document = file.read() # represents the content of the file and puts it into the return_policy_document

    headers_to_split_on = [('#', "Header 1"),('##', "Header 2")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False) # Makes sure that the headers don't get stripped off
    documents = markdown_splitter.split_text(policy_document)

    for id, document in enumerate(documents):#numbers each document with an id number
        document.id = str(id + 1) 

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=gemini_embeddings,
        persist_directory='./policy_embeddings'
    )

    # query = 'my plant was dead when it arrived'

    # results = vector_store.similarity_search(query, k=2)

    # for result in results:
    #     print(result)

    #
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3}) # Finds the 3 most 'interesting"
    return retriever

def create_rag_chain(retriever):

    rag_prompt = PromptTemplate( 
        template="""Answer the customer's questions about return and refunds using only the 
        context provided. If the context can't be used to answer the customer's question, reply "I don't know'


        Use this context to answer the question:
        {context}

        Customer's Question:
        {question}

        Response:
        """,
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.1) # Makes the LLM keep non-creative about the policy. 

    plant_rag_chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return plant_rag_chain

@tool
def return_rag(customer_query):
    "answer customer questions about returns and refunds"
    return refund_rag_chain.invoke(customer_query)


retriever = create_reciever()
refund_rag_chain = create_rag_chain(retriever)

# query = "my plant was unhappy when it arrived"

# response = refund_rag_chain.invoke(query)

# print(response)