import streamlit as st
import os
import pandas as pd
import pickle
import openai
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from scipy import spatial

# from openai import OpenAI
import logging
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
chatOpenAiKey = os.getenv('CHAT_OPENAI_KEY')
prompt = ChatPromptTemplate.from_template(""""You are a ChatBot eager to encourage people to visits this city you are 
very excited about.":

        <context>
        {context}
        </context>

        Question: {input}""")

no_matches = "Sorry I looked but could not find that just now."


def distances_from_embeddings(
        query_embedding: List[float],
        embeddings: List[List[float]],
        distance_metric="cosine",
) -> List[List]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def create_context(question, df, max_len=1800, size="ada"):
    logging.info("Creating context for the question.")
    try:
        # Get the embeddings for the question
        q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0][
            'embedding']
        logging.info("Obtained embeddings for the question.")
    except Exception as e:
        logging.error(f"Error obtaining embeddings for the question: {e}")
        return ""

    try:
        # Get the distances from the embeddings
        df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
        logging.info("Calculated distances from embeddings.")
    except Exception as e:
        logging.error(f"Error calculating distances from embeddings: {e}")
        return ""

    returns = []
    cur_len = 0

    try:
        # Sort by distance and add the text to the context until the context is too long
        for i, row in df.sort_values('distances', ascending=True).iterrows():
            cur_len += row['n_tokens'] + 4
            if cur_len > max_len:
                logging.info("Reached max length for context.")
                break
            returns.append(row["text"])
        logging.info("Context creation complete.")
    except Exception as e:
        logging.error(f"Error during context creation: {e}")
        return ""

    return "\n\n###\n\n".join(returns)


class WebPage:
    def __init__(self):

        self.llm_connection = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            api_key=chatOpenAiKey,
            streaming=True,
            openai_organization=None,
            tiktoken_model_name=None,
            default_headers=None,
            default_query=None,
            http_client=None,
            http_async_client=None
        )
        self.current_query = None

    def answer_question(self, df, model="gpt-3.5-turbo", question="What is there to do here?", max_len=1800,
                        size="ada",
                        debug=False, max_tokens=1500, stop_sequence=None):
        logging.info(f"Answering question: {question}")
        conversation_context = " ".join(
            [f"Question: {q}, Answer: {a}" for q, a in st.session_state.conversation_history] + [
                f"Question: {question}"])

        context = create_context(question, df, max_len=max_len, size=size)
        if context == "":
            logging.error("Context creation failed.")
            return "I couldn't generate context for the question."
        if debug:
            logging.info(f"Context:\n{context}\n\n")

        document_chain = create_stuff_documents_chain(self.llm_connection, prompt)
        context_document = Document(page_content=context)
        self.current_query = ""
        logging.info(
            f"*****************************************    Asking LLM: {prompt} {question}. "
            f"Here is the conversation history:  {conversation_context}")
        return document_chain.invoke({
            "input": f"{prompt} {question}. Here is the conversation history:  {conversation_context}",
            "context": [context_document]
        })

    def main(self):
        logging.info("Loading the DataFrame.")
        try:
            with open('df.pkl', 'rb') as f:
                df = pickle.load(f)
            logging.info("DataFrame loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading the DataFrame: {e}")
        st.title('Q&A App about Hartford Wisconsin')

        question = st.text_input("Ask a question:")

        if st.button('Get Answer'):
            logging.info(f"Button clicked with question: {question}")
            with st.spinner('Generating answer...'):
                try:
                    context = create_context(question, df)
                    if context:
                        logging.info(f"Generated context: {context}")
                        answer = self.answer_question(df, question=question)
                        logging.info(f"Generated answer: {answer}")
                        st.write(answer)
                        logging.info("Displayed answer to the user.")
                    else:
                        st.write("Context creation failed.")
                        logging.error("Context creation failed.")
                except Exception as e:
                    logging.error(f"Error in generating answer: {e}")
                    st.write("There was an error generating the answer.")


web_page = WebPage()

# Call the main method on the web_page instance
if __name__ == "__main__":
    web_page.main()
