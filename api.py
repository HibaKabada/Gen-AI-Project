from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from contextlib import asynccontextmanager
import shutil
import uuid
import tempfile
import pandas as pd


from vector import (
    connect_to_cloud_sql,
    fetch_data_from_table,
    create_vector_store_from_sql_data,
    retrieve_relevant_documents_with_scores,
    generate_response_with_rag,
    analyze_infographic,
    get_esg_insights,


    save_feedback,
    get_feedback_data,
    get_average_rating,
    generate_dashboard_insights

)

class Query(BaseModel):
    text: str
    top_k: Optional[int] = 3
    model: Optional[str] = "gemini-pro"
    language: Optional[str] = "English" 

class Document(BaseModel):
    content: str
    metadata: dict
    score: Optional[float] = None

class Response(BaseModel):
    answer: str
    sources: Optional[List[Document]] = None

class InsightResponse(BaseModel):
    insights: str


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    try:
        print("🔎 Chargement des données depuis Cloud SQL...")
        documents = fetch_data_from_table()
        
        if not documents:
            print("⚠️ Aucune donnée trouvée dans la table.")
        else:
            print("📊 Création/chargement de l'index FAISS...")
            embeddings_model_name = "textembedding-gecko@latest"
            vector_store = create_vector_store_from_sql_data(documents, embeddings_model_name)
            print("✅ Vector store chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du vector store: {e}")
    
    yield
    
    print("🧹 Nettoyage des ressources...")
    for file in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

app = FastAPI(
    title="API de RAG avec Gemini",
    description="API pour un système de Retrieval Augmented Generation utilisant Gemini et FAISS",
    version="1.0.0",
    lifespan=lifespan
)


def get_vector_store():
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Le vector store n'est pas disponible")
    return vector_store


@app.post("/refresh", tags=["administration"])
async def refresh_vector_store():
    """
    Rafraîchit le vector store en rechargeant les données depuis la base de données.
    """
    global vector_store
    try:
        documents = fetch_data_from_table()
        if not documents:
            raise HTTPException(status_code=404, detail="Aucune donnée trouvée dans la table")
        
        embeddings_model_name = "textembedding-gecko@latest"
        vector_store = create_vector_store_from_sql_data(documents, embeddings_model_name)
        return {"status": "success", "message": "Vector store rafraîchi avec succès", "document_count": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du rafraîchissement du vector store: {str(e)}")


@app.post("/query", response_model=Response, tags=["rag"])
async def query_rag(query: Query, vs=Depends(get_vector_store)):
    """
    Effectue une requête RAG (Retrieval Augmented Generation).
    """
    try:
        relevant_docs, scores = retrieve_relevant_documents_with_scores(query.text, vs, query.top_k)
        
        system_instruction = ""
        if query.language == "Français":
            system_instruction = "Réponds toujours en français, quelle que soit la langue de la question."
        elif query.language == "Arabic":
            system_instruction = "أجب دائمًا باللغة العربية، بغض النظر عن لغة السؤال."
            
        modified_query = query.text
        if system_instruction:
            modified_query = f"{system_instruction}\n\n{query.text}"
            
        response = generate_response_with_rag(modified_query, vs, query.model)
        
        sources = []
        for i, doc in enumerate(relevant_docs):
            sources.append(Document(
                content=doc.page_content,
                metadata=doc.metadata,
                score=scores[i]
            ))
        
        return Response(
            answer=response.content,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse: {str(e)}")


@app.post("/similar-documents", tags=["retrieval"])
async def get_similar_documents(query: Query, vs=Depends(get_vector_store)):
    """
    Récupère les documents les plus similaires à la requête sans génération de réponse.
    """
    try:
        relevant_docs, scores = retrieve_relevant_documents_with_scores(query.text, vs, query.top_k)
        
        result = []
        for i, doc in enumerate(relevant_docs):
            result.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": scores[i]
            })
        
        return {"status": "success", "documents": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche de documents: {str(e)}")

@app.post("/analyze-image", response_model=InsightResponse, tags=["vision"])
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an infographic or image related to ESG topics using Gemini Pro Vision.
    """
    try:
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        insights = analyze_infographic(file_path)
        
        return InsightResponse(insights=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.post("/esg-insights", response_model=InsightResponse, tags=["insights"])
async def generate_esg_insights(query: Query):
    """
    Generate ESG insights using Gemini Pro without RAG.
    """
    try:
        insights = get_esg_insights(query.text)
        return InsightResponse(insights=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@app.get("/health", tags=["système"])
async def health_check():
    """
    Vérifie l'état de santé de l'API.
    """
    return {
        "status": "healthy",
        "vector_store_loaded": vector_store is not None
    }


class Feedback(BaseModel):
    question: str
    model_answer: str
    rating: float
    comments: Optional[str] = None

@app.post("/submit-feedback", tags=["feedback"])

async def submit_feedback(feedback: Feedback):

    """
    Submit user feedback about the model's response.
    """
    try:
        save_feedback(feedback.question, feedback.model_answer, feedback.rating, feedback.comments)
        return {"status": "success", "message": "Feedback submitted successfully!"}

    except Exception as e:
        print(f"Error saving feedback: {str(e)}")  
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


@app.get("/feedback-data", tags=["feedback"])
async def get_feedback():
    """
    Retrieve all feedback data for the dashboard.
    """
    try:
        feedback_df = get_feedback_data()
        if feedback_df.empty:
            return []
        try:
            records = feedback_df.to_dict(orient="records")
            return records
        except Exception as e:
            print(f"Error converting DataFrame to records: {str(e)}")
            return []
    except Exception as e:
        print(f"Error retrieving feedback data: {str(e)}")
        return {"status": "error", "message": f"Error retrieving feedback data: {str(e)}"}

 
@app.get("/feedback-stats", tags=["feedback"])
async def get_feedback_stats():
    """
    Retrieve feedback statistics for the dashboard.
    """
    try:
        avg_rating = get_average_rating()
        return {
            "average_rating": avg_rating
        }
    except Exception as e:
        print(f"Error retrieving feedback stats: {str(e)}")  
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback stats: {str(e)}")


@app.get("/generate-insights", tags=["feedback"])
async def generate_insights():
    """
    Generate insights from feedback data using the Gemini API.
    """
    try:
        feedback_df = get_feedback_data()
        if feedback_df.empty:
            return {"status": "success", "insights": "No feedback data available."}
        try:
            insights = generate_dashboard_insights(feedback_df)
            return {"status": "success", "insights": insights}
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            return {"status": "error", "insights": f"Error generating insights: {str(e)}"}

    except Exception as e:
        print(f"Error accessing feedback data: {str(e)}")
        return {"status": "error", "insights": f"Error accessing feedback data: {str(e)}"}
    

if __name__ == "__main__":

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)