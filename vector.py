import os 
from dotenv import load_dotenv
import pymysql
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_core.documents import Document
from google.cloud.sql.connector import Connector, IPTypes
import sys
from config import PROJECT_ID, REGION, INSTANCE, DATABASE, DB_USER, TABLE_NAME
import pg8000
from sqlalchemy import create_engine, text
import google.generativeai as genai
from PIL import Image
import io
import google.generativeai as genai
import pandas as pd

load_dotenv()


api_key = os.getenv('GEMINI_API_KEY')
db_password = os.getenv('DB_PASSWORD')


# Feedback CSV file path
FEEDBACK_CSV = "feedback.csv"


if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

if not db_password:
    raise ValueError("DB_PASSWORD is missing. Check your .env file.")


genai.configure(api_key=api_key)

def connect_to_cloud_sql():
    connector = Connector()
    
    def getconn():
        return connector.connect(
            f"{PROJECT_ID}:{REGION}:{INSTANCE}",
            "pg8000",
            user=DB_USER,
            password=db_password,
            db=DATABASE
        )
    
    engine = create_engine(
        "postgresql+pg8000://",
        creator=getconn
    )
    
    return engine

def fetch_data_from_table():
    connector = Connector()

    def getconn():
        return connector.connect(
            f"{PROJECT_ID}:{REGION}:{INSTANCE}",
            "pg8000",
            user=DB_USER,
            password=db_password,
            db=DATABASE,
        )

    conn = getconn()
    cursor = conn.cursor()

    try:
        cursor.execute(f"SELECT * FROM {TABLE_NAME}")
        rows = cursor.fetchall()
        documents = []
        for i, row in enumerate(rows):
            text_content = row[1] 
            doc = Document(page_content=text_content, metadata={"id": i})
            documents.append(doc)

        return documents
    except Exception as e:
        print(f"Error in fetch_data_from_table: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


def create_vector_store_from_sql_data(documents, embeddings_model_name):
    """
    Create a fresh vector store from the SQL text data using the provided embedding model.
    """
    embeddings = VertexAIEmbeddings(model_name=embeddings_model_name)

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

    vector_store.save_local('faiss_index')

    return vector_store

def retrieve_relevant_documents(query, vector_store, top_k=3):
    """
    Retrieve relevant documents from the vector store using the query.
    """
    return vector_store.similarity_search(query, k=top_k)

def retrieve_relevant_documents_with_scores(query, vector_store, top_k=3):
    """
    Retrieve relevant documents from the vector store using the query,
    and return both documents and their similarity scores.
    """
    docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    docs = [doc for doc, score in docs_with_scores]
    #pour faiss, les scores sont en fait des distances, donc on les convertit en scores de similarité
    #faiss retourne la distance L2, qui est plus faible pour les éléments plus similaires
    #on convertit en similarité cosinus, qui est plus élevée (proche de 1) pour les éléments plus similaires

    scores = [1 - (score / 2) for doc, score in docs_with_scores]  # convertir en 0-1 score de similarité
    
    return docs, scores

def analyze_infographic(image_path):
    """
    Analyze an infographic using Gemini Pro Vision.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Text string with analysis of the infographic
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # convertir image en objet PIL
        image = Image.open(io.BytesIO(image_data))
        
        #générer contenu 
        response = model.generate_content(["Analyze this infographic and provide key ESG insights and sustainability recommendations.", image])
        return response.text
    except Exception as e:
        print(f"Error analyzing infographic: {str(e)}")
        return f"Error analyzing image: {str(e)}"

def get_esg_insights(query):
    """
    Get ESG insights using Gemini Pro.
    
    Args:
        query: Text query about ESG topics
        
    Returns:
        Text string with ESG insights
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        print(f"Error getting ESG insights: {str(e)}")
        return f"Error generating insights: {str(e)}"

def save_feedback(question: str, model_answer: str, rating: int, comments: str = None):
    """
    Save user feedback to a CSV file.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feedback_data = {
            "Timestamp": timestamp,
            "Question": question,
            "Model Answer": model_answer,
            "Rating": rating,
            "Comments": comments if comments else ""
        }

        if not os.path.exists(FEEDBACK_CSV):
            feedback_df = pd.DataFrame(columns=["Timestamp", "Question", "Model Answer", "Rating", "Comments"])
        else:
            try:
                feedback_df = pd.read_csv(FEEDBACK_CSV)
            except Exception as e:
                print(f"Error reading feedback CSV: {str(e)}")
                feedback_df = pd.DataFrame(columns=["Timestamp", "Question", "Model Answer", "Rating", "Comments"])

        new_row_df = pd.DataFrame([feedback_data])
        feedback_df = pd.concat([feedback_df, new_row_df], ignore_index=True)
        feedback_df.to_csv(FEEDBACK_CSV, index=False)
        return True
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False


def get_feedback_data():
    """
    Retrieve all feedback data from the CSV file.
    """
    try:
        if os.path.exists(FEEDBACK_CSV):
            feedback_df = pd.read_csv(FEEDBACK_CSV)
            return feedback_df
        else:
            return pd.DataFrame(columns=["Timestamp", "Question", "Model Answer", "Rating", "Comments"])
    except Exception as e:
        print(f"Error retrieving feedback data: {str(e)}")
        return pd.DataFrame(columns=["Timestamp", "Question", "Model Answer", "Rating", "Comments"])


def get_average_rating():
    """
    Calculate the average rating from feedback data.
    """
    try:
        feedback_df = get_feedback_data()
        if not feedback_df.empty and "Rating" in feedback_df.columns:
            return float(feedback_df["Rating"].mean())
        return 0.0
    except Exception as e:
        print(f"Error calculating average rating: {str(e)}")
        return 0.0

def generate_dashboard_insights(feedback_data: pd.DataFrame):
    """
    Generate insights and summaries from feedback data using the Gemini API.
    """
    try:
        if feedback_data.empty:
            return "No feedback data available for analysis."

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return "Gemini API key not configured. Unable to generate insights."


        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash') 

        total_feedback = len(feedback_data)
        avg_rating = feedback_data["Rating"].mean() if "Rating" in feedback_data.columns else 0


        sample_size = min(50, total_feedback) 
        sampled_data = feedback_data.sample(sample_size) if total_feedback > sample_size else feedback_data


        feedback_summary = f"Total feedback entries: {total_feedback}\n"
        feedback_summary += f"Average rating: {avg_rating:.2f}/5\n\n"
        feedback_summary += "Sample of feedback entries:\n"


        for idx, row in sampled_data.iterrows():
            try:
                rating = row["Rating"] if "Rating" in row else "N/A"
                comments = row["Comments"] if "Comments" in row and not pd.isna(row["Comments"]) else "No comment"
                feedback_summary += f"- Rating: {rating}, Comment: {comments}\n"
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue


        prompt = f"""

        Analyze the following feedback data about an ESG AI assistant and provide insights:

        1. Summarize the overall sentiment.

        2. Highlight common themes in the comments.

        3. Suggest improvements based on the feedback.

        4. Provide a brief summary of the average rating.

 
        Feedback Data Summary:

        {feedback_summary}

        """
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error in Gemini API call: {str(e)}")
            return f"Unable to generate AI insights. Basic statistics: {total_feedback} feedback entries with average rating of {avg_rating:.2f}/5."
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        return f"Error analyzing feedback data: {str(e)}"


def generate_response_with_rag(query, vector_store, generation_model="gemini-pro"):
    """
    Use RAG logic: retrieve relevant documents and generate a response.
    """
    #récupérer les documents pertinents
    relevant_docs = retrieve_relevant_documents(query, vector_store)

    if not relevant_docs:
        return "No relevant information found."

    #formater le contexte à partir des documents récupérés
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    #creer une prompt pour le rag
    rag_prompt = f"""
    Answer the following question based on the provided context.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """

    #utiliser modele gemini pour générer une réponse
    chat_model = ChatVertexAI(model_name=generation_model)
    response = chat_model.invoke(rag_prompt)

    return response

if __name__ == "__main__":
    try:
        #récupérer les données texte de cloud sql en tant qu'objets Document
        print("🔎 Fetching data from Cloud SQL...")
        documents = fetch_data_from_table()

        if not documents:
            print("❌ No data found in the table.")
            exit(1)

        #créér vector store
        embeddings_model_name = "textembedding-gecko@latest"
        print("📊 Creating/loading FAISS index...")
        vector_store = create_vector_store_from_sql_data(documents, embeddings_model_name)

        query = "What is the impact of AI on business?"
        print(f"🗣️ Query: {query}")
        
        response = generate_response_with_rag(query, vector_store)
        print(f"✅ Response: {response.content}")

    except Exception as e:
        print(f"❌ Error: {e}")