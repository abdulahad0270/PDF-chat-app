import streamlit as st
import pickle 
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PyPDF2 import PdfFileReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title('PDF chat app')
    st.markdown("""

     This app is built using <br>
     -[Streamlit](https://streamlit.io/)<br>
     -[LangChain](https://www.langchain.com/)<br>
     -[openAI](https://openai.com/blog/openai-api)
    """,unsafe_allow_html=True)
    add_vertical_space(5)
    st.write("Made by Abdul Ahad")

def main():
    st.header("PDF Chatbot")

    load_dotenv()

     #upload pdf
    pdf=st.file_uploader("Upload your PDF", type = 'pdf')
    st.write(pdf.name)

    #Reading the pdf
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)

        #Reading one page at a time
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()       
            
        #Now Splitting the text as per model requirements
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        #Converting the text to chunk
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks)

        #Text Embedding
        """
        
    Text embedding refers to the process of converting textual data into numerical vectors, 
    which can be used as inputs for machine learning models, natural language processing (NLP) tasks, and various text analysis applications"""
        
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"(store_name).pkl"):
            with open(f"(store_name).pkl","rb") as f:
                VectorStore = pickle.load(f)
            #st.write("Embeddings Loaded from the Disk")
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding = embeddings)
            with open(f"(store_name).pkl","wb") as f:
                pickle.dump(vectorstore, f)
            #st.write('Embedding Computation Completed')

        #Accept user question
        query = st.text_input("Ask Question about your pdf")
        #st.write(query)

        if query:
            docs = VectorStore.similarity_search(query = query , k =3)
            llm = OpenAI(temperature = 0,)
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_document = docs, question = query)
                print(cb)
            st.write(response)

            #st.write(docs)




            
            



if __name__ == '__main__':
    main()

