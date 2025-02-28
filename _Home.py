import streamlit as st
import requests
import os
import tempfile  
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

HOST = "http://127.0.0.1:8181"


try:
    api_test = requests.get(f"{HOST}/health", timeout=2)
    if api_test.status_code == 200:
        print("API is available")
    else:
        print(f"API returned unexpected status: {api_test.status_code}")
except Exception as e:
    print(f"Cannot connect to API: {str(e)}")



st.set_page_config(page_title="🌱 ESG AI Strategist", page_icon="🌍", layout="wide")

st.title("🌱 ESG AI Strategist")
st.markdown("##### Revolutionizing sustainable futures with AI-powered insights! 💡")


interface_text = {
    "English": {
        "sidebar_title": "🌿 ESG AI Assistant",
        "sidebar_desc": "Your guide to sustainability with AI-powered insights 🚀",
        "input_placeholder": "Enter your ESG query...",
        "generating": "Generating response...",
        "show_sources": "📚 Show sources",
        "show_similar": "🔍 Show similar documents",
        "last_sources": "📚 Show sources from last query",
        "last_similar": "🔍 Show similar documents from last query",
        "sources_title": "📚 Sources",
        "similar_title": "🔍 Similar Documents",
        "refresh_button": "🔄 Refresh Vector Store",
        "refreshing": "Refreshing vector store...",
        "refresh_success": "Vector store refreshed successfully! ({} documents loaded)",
        "connect_error": "Error: Unable to connect to the API server. Make sure it's running and accessible.",
        "timeout_error": "Error: The API request timed out. Please try again.",
        "unexpected_error": "Error: An unexpected error occurred: {}",
        "health_ok": "✅ API is healthy and vector store is loaded",
        "health_warning": "⚠️ API is healthy but vector store is not loaded",
        "health_error": "❌ API health check failed",
        "connect_fail": "❌ Cannot connect to API",
        "image_tab": "📊 Image Analysis",
        "text_tab": "💬 Text Chat",
        "upload_image": "Upload an ESG infographic or sustainability report",
        "analyze_image": "Analyze Image",
        "analyzing_image": "Analyzing your image...",
        "image_result": "Analysis Results",
        "no_image": "Please upload an image first.",
        "image_error": "Error analyzing image: {}",
        "feedback_tab": "📝 Feedback Form",  
        "dashboard_tab": "📊 Feedback Dashboard",  
        "feedback_title": "Feedback Form",  
        "feedback_desc": "Help us improve by providing feedback on the model's answers.", 
        "feedback_question": "Your Question",  
        "feedback_answer": "Model's Answer",  
        "feedback_rating": "Rate the Model's Answer (1 = Poor, 5 = Excellent)", 
        "feedback_comments": "Additional Comments (Optional)",  
        "feedback_submit": "Submit Feedback",  
        "feedback_success": "Thank you for your feedback!",  
        "dashboard_title": "Feedback Dashboard", 
        "dashboard_desc": "View feedback and insights from users.",  
        "dashboard_avg_rating": "Average Rating",  
        "dashboard_all_feedback": "All Feedback",  
        "dashboard_insights": "Insights",  
        "dashboard_common_comments": "Most Common Comments",  
    },
    "Français": {
        "sidebar_title": "🌿 Assistant IA ESG",
        "sidebar_desc": "Votre guide pour la durabilité avec des insights alimentés par l'IA 🚀",
        "input_placeholder": "Entrez votre question ESG...",
        "generating": "Génération de la réponse...",
        "show_sources": "📚 Afficher les sources",
        "show_similar": "🔍 Afficher les documents similaires",
        "last_sources": "📚 Afficher les sources de la dernière requête",
        "last_similar": "🔍 Afficher les documents similaires de la dernière requête",
        "sources_title": "📚 Sources",
        "similar_title": "🔍 Documents similaires",
        "refresh_button": "🔄 Rafraîchir le Vector Store",
        "refreshing": "Rafraîchissement du vector store...",
        "refresh_success": "Vector store rafraîchi avec succès ! ({} documents chargés)",
        "connect_error": "Erreur : Impossible de se connecter au serveur API. Assurez-vous qu'il est en cours d'exécution et accessible.",
        "timeout_error": "Erreur : La requête API a expiré. Veuillez réessayer.",
        "unexpected_error": "Erreur : Une erreur inattendue s'est produite : {}",
        "health_ok": "✅ L'API est fonctionnelle et le vector store est chargé",
        "health_warning": "⚠️ L'API est fonctionnelle mais le vector store n'est pas chargé",
        "health_error": "❌ Vérification de l'état de l'API échouée",
        "connect_fail": "❌ Impossible de se connecter à l'API",
        "image_tab": "📊 Analyse d'Image",
        "text_tab": "💬 Chat Textuel",
        "upload_image": "Téléchargez une infographie ESG ou un rapport de durabilité",
        "analyze_image": "Analyser l'Image",
        "analyzing_image": "Analyse de votre image...",
        "image_result": "Résultats de l'Analyse",
        "no_image": "Veuillez d'abord télécharger une image.",
        "image_error": "Erreur lors de l'analyse de l'image : {}",
        "feedback_tab": "📝 Formulaire de Feedback",  
        "dashboard_tab": "📊 Tableau de Bord de Feedback",  
        "feedback_title": "Formulaire de Feedback",  
        "feedback_desc": "Aidez-nous à nous améliorer en fournissant des commentaires sur les réponses du modèle.",  
        "feedback_question": "Votre Question",  
        "feedback_answer": "Réponse du Modèle",  
        "feedback_rating": "Notez la Réponse du Modèle (1 = Médiocre, 5 = Excellent)",  
        "feedback_comments": "Commentaires Supplémentaires (Optionnel)",  
        "feedback_submit": "Soumettre le Feedback",  
        "feedback_success": "Merci pour vos commentaires !",  
        "dashboard_title": "Tableau de Bord de Feedback",  
        "dashboard_desc": "Affichez les commentaires et les insights des utilisateurs.",  
        "dashboard_avg_rating": "Note Moyenne",  
        "dashboard_all_feedback": "Tous les Commentaires", 
        "dashboard_insights": "Insights",  
        "dashboard_common_comments": "Commentaires les Plus Courants",  
    },
    "Arabic": {
        "sidebar_title": "🌿 مساعد ESG الذكي",
        "sidebar_desc": "دليلك للاستدامة مع رؤى مدعومة بالذكاء الاصطناعي 🚀",
        "input_placeholder": "أدخل استفسارك عن ESG...",
        "generating": "جاري إنشاء الرد...",
        "show_sources": "📚 عرض المصادر",
        "show_similar": "🔍 عرض المستندات المشابهة",
        "last_sources": "📚 عرض مصادر الاستعلام الأخير",
        "last_similar": "🔍 عرض المستندات المشابهة للاستعلام الأخير",
        "sources_title": "📚 المصادر",
        "similar_title": "🔍 المستندات المشابهة",
        "refresh_button": "🔄 تحديث مخزن الفيكتور",
        "refreshing": "جاري تحديث مخزن الفيكتور...",
        "refresh_success": "تم تحديث مخزن الفيكتور بنجاح! ({} مستندات محملة)",
        "connect_error": "خطأ: تعذر الاتصال بخادم API. تأكد من أنه قيد التشغيل ويمكن الوصول إليه.",
        "timeout_error": "خطأ: انتهت مهلة طلب API. يرجى المحاولة مرة أخرى.",
        "unexpected_error": "خطأ: حدث خطأ غير متوقع: {}",
        "health_ok": "✅ API بصحة جيدة ومخزن الفيكتور محمل",
        "health_warning": "⚠️ API بصحة جيدة ولكن مخزن الفيكتور غير محمل",
        "health_error": "❌ فشل فحص حالة API",
        "connect_fail": "❌ لا يمكن الاتصال بـ API",
        "image_tab": "📊 تحليل الصورة",
        "text_tab": "💬 الدردشة النصية",
        "upload_image": "قم بتحميل رسم بياني أو تقرير استدامة متعلق بـ ESG",
        "analyze_image": "تحليل الصورة",
        "analyzing_image": "جاري تحليل الصورة...",
        "image_result": "نتائج التحليل",
        "no_image": "يرجى تحميل صورة أولا.",
        "image_error": "خطأ في تحليل الصورة: {}",
        "feedback_tab": "📝 نموذج التقييم",  
        "dashboard_tab": "📊 لوحة تحليل التقييمات",  
        "feedback_title": "نموذج التقييم",  
        "feedback_desc": "ساعدنا في التحسين من خلال تقديم تعليقاتك على إجابات النموذج.",  
        "feedback_question": "سؤالك",  
        "feedback_answer": "إجابة النموذج",  
        "feedback_rating": "قيم إجابة النموذج (1 = ضعيف, 5 = ممتاز)",  
        "feedback_comments": "تعليقات إضافية (اختياري)",  
        "feedback_submit": "إرسال التقييم",  
        "feedback_success": "شكرًا على تعليقاتك!", 
        "dashboard_title": "لوحة تحليل التقييمات",  
        "dashboard_desc": "عرض التقييمات والرؤى من المستخدمين.",  
        "dashboard_avg_rating": "متوسط التقييم",  
        "dashboard_all_feedback": "جميع التقييمات",  
        "dashboard_insights": "الرؤى",  
        "dashboard_common_comments": "أكثر التعليقات شيوعًا",  
    },
}


with st.sidebar:
    language = st.selectbox("Language", ["English", "Français", "Arabic"])

texts = interface_text.get(language, interface_text["English"])

with st.sidebar:
    st.title(texts["sidebar_title"])
    st.write(texts["sidebar_desc"])

temperature = 0.2
top_k = 3
model = "gemini-pro"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": texts["input_placeholder"]}]

if "sources_data" not in st.session_state:
    st.session_state["sources_data"] = None

if "similar_docs_data" not in st.session_state:
    st.session_state["similar_docs_data"] = None

if "image_analysis" not in st.session_state:
    st.session_state["image_analysis"] = None

def display_similarity_bar(score):

    percentage = int(score * 100)
    
    if score < 0.5:
        color = "red"
    elif score < 0.7:
        color = "orange"
    elif score < 0.9:
        color = "lightgreen"
    else:
        color = "green"
    
    st.write(f"**Similarity Score:** {percentage}%")
    st.progress(score, text="")

    st.markdown(f"""
    <div style="
        width: {percentage}%; 
        height: 10px; 
        background-color: {color}; 
        border-radius: 5px;
        margin-bottom: 10px;
    "></div>
    """, unsafe_allow_html=True)

def display_sources(sources):
    if sources:
        tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])
        
        for i, (tab, source) in enumerate(zip(tabs, sources)):
            with tab:
                if "score" in source:
                    display_similarity_bar(source["score"])
                
                st.write("**Metadata:**")
                st.json(source.get("metadata", {}))
                st.write("**Content:**")
                st.write(source.get("content", "No content available."))

def display_similar_docs(documents):
    if documents:
        doc_tabs = st.tabs([f"Document {i+1}" for i in range(len(documents))])
        
        for i, (tab, doc) in enumerate(zip(doc_tabs, documents)):
            with tab:
                if "score" in doc:
                    display_similarity_bar(doc["score"])
                    
                st.write("**Metadata:**")
                st.json(doc.get("metadata", {}))
                st.write("**Content:**")
                st.write(doc.get("content", "No content available."))

tab1, tab2, tab3, tab4 = st.tabs([
    texts["text_tab"], texts["image_tab"], texts["feedback_tab"], texts["dashboard_tab"]
])

with tab1:
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "🧑‍💻"
        st.chat_message(message["role"], avatar=avatar).write(message["content"])

    if user_input := st.chat_input(texts["input_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user", avatar="🧑‍💻").write(user_input)

        with st.spinner(texts["generating"]):
            query_payload = {
                "text": user_input,
                "top_k": top_k,
                "model": model,
                "language": language 
            }
            
            try:
                response = requests.post(
                    f"{HOST}/query",
                    json=query_payload,
                    timeout=60
                )
                

                documents_response = requests.post(
                    f"{HOST}/similar-documents",
                    json=query_payload,
                    timeout=30
                )
            
                if response.status_code == 200:
                    response_data = response.json()
                    answer = response_data.get("answer", "No response received.")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.chat_message("assistant", avatar="🤖").write(answer)

                    if "sources" in response_data and response_data["sources"]:
                        st.session_state["sources_data"] = response_data["sources"]
                        
                        if st.button(texts["show_sources"], key="show_sources"):
                            st.subheader(texts["sources_title"])
                            display_sources(st.session_state["sources_data"])
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.write(f"Details: {response.text}")

                if documents_response.status_code == 200:
                    documents_data = documents_response.json()
                    if "documents" in documents_data and documents_data["documents"]:
                        st.session_state["similar_docs_data"] = documents_data["documents"]
                        
                        if st.button(texts["show_similar"], key="show_similar_docs"):
                            st.subheader(texts["similar_title"])
                            display_similar_docs(st.session_state["similar_docs_data"])
                
            except requests.exceptions.ConnectionError:
                st.error(texts["connect_error"])
            except requests.exceptions.Timeout:
                st.error(texts["timeout_error"])
            except Exception as e:
                st.error(texts["unexpected_error"].format(str(e)))

    if st.session_state.get("sources_data"):
        if st.button(texts["last_sources"], key="show_last_sources"):
            st.subheader(texts["sources_title"])
            display_sources(st.session_state["sources_data"])

    if st.session_state.get("similar_docs_data"):
        if st.button(texts["last_similar"], key="show_last_similar_docs"):
            st.subheader(texts["similar_title"])
            display_similar_docs(st.session_state["similar_docs_data"])


with tab2:
    st.header("ESG Infographic & Report Analyzer")
    st.write("Upload infographics, sustainability reports, ESG charts or other visual content for AI analysis.")
    
    uploaded_file = st.file_uploader(texts["upload_image"], type=["jpg", "jpeg", "png", "pdf"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except:
            st.write("Uploaded file is not an image or cannot be displayed")
        
        if st.button(texts["analyze_image"]):
            with st.spinner(texts["analyzing_image"]):
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
                    temp_file.write(uploaded_file.getvalue())
                    temp_file.close()

                    files = {"file": (uploaded_file.name, open(temp_file.name, "rb"), f"image/{os.path.splitext(uploaded_file.name)[1][1:]}")}

                    response = requests.post(
                        f"{HOST}/analyze-image",
                        files=files,
                        timeout=60
                    )

                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        analysis_result = response_data.get("insights", "No analysis results received.")

                        st.session_state["image_analysis"] = analysis_result
                        st.subheader(texts["image_result"])
                        st.write(analysis_result)
                    else:
                        st.error(f"Error: API returned status code {response.status_code}")
                        st.write(f"Details: {response.text}")
                    
                except requests.exceptions.ConnectionError:
                    st.error(texts["connect_error"])
                except requests.exceptions.Timeout:
                    st.error(texts["timeout_error"])
                except Exception as e:
                    st.error(texts["image_error"].format(str(e)))
    else:
        if st.button(texts["analyze_image"]):
            st.warning(texts["no_image"])

FEEDBACK_CSV = "feedback.csv"

if not os.path.exists(FEEDBACK_CSV):
    feedback_df = pd.DataFrame(columns=["Question", "Model Answer", "Rating"])
    feedback_df.to_csv(FEEDBACK_CSV, index=False)


def generate_feedback_dashboard(feedback_df):
    """
    Génère un histogramme pour le dashboard de feedback.
    """
    if feedback_df.empty:
        st.warning("No feedback data available.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(feedback_df["Rating"], bins=5, range=(1, 6), color='skyblue', edgecolor='black')
    ax.set_title("Distribution des Notes", fontsize=14)
    ax.set_xlabel("Note", fontsize=12)
    ax.set_ylabel("Nombre de Feedback", fontsize=12)
    ax.set_xticks(range(1, 6))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig)



with tab3:  
    st.header(texts["feedback_title"])
    st.write(texts["feedback_desc"])

    with st.form("feedback_form"):
        question = st.text_input(texts["feedback_question"])
        model_answer = st.text_area(texts["feedback_answer"])
        rating = st.slider(texts["feedback_rating"], 1, 5)
        submitted = st.form_submit_button(texts["feedback_submit"])

        if submitted:
            try:
                feedback_payload = {
                    "question": question,
                    "model_answer": model_answer,
                    "rating": rating
                    #"comments": comments
                }
                response = requests.post(f"{HOST}/submit-feedback", json=feedback_payload)
                if response.status_code == 200:
                    st.success(texts["feedback_success"])
                else:
                    st.error(f"Error submitting feedback: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


with tab4:  
    st.header(texts["dashboard_title"])
    st.write(texts["dashboard_desc"])
    try:

        feedback_df = pd.read_csv(FEEDBACK_CSV)

        st.subheader("Statistiques de Base")
        st.write(f"Nombre total de feedbacks : {len(feedback_df)}")
        st.write(f"Note moyenne : {feedback_df['Rating'].mean():.2f} ⭐")

        st.subheader("Visualisations des Feedbacks")
        generate_feedback_dashboard(feedback_df)
        st.subheader(texts["dashboard_all_feedback"])
        st.dataframe(feedback_df)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de feedback : {str(e)}")


with st.sidebar:
    st.subheader("System Status")
    
    if st.button(texts["refresh_button"]):
        with st.spinner(texts["refreshing"]):
            try:
                response = requests.post(f"{HOST}/refresh", timeout=30)
                
                if response.status_code == 200:
                    response_data = response.json()
                    document_count = response_data.get("document_count", 0)
                    st.success(texts["refresh_success"].format(document_count))
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error(texts["connect_error"])
            except requests.exceptions.Timeout:
                st.error(texts["timeout_error"])
            except Exception as e:
                st.error(texts["unexpected_error"].format(str(e)))
    

    try:
        health_response = requests.get(f"{HOST}/health", timeout=5)
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data.get("status") == "healthy":
                if health_data.get("vector_store_loaded"):
                    st.success(texts["health_ok"])
                else:
                    st.warning(texts["health_warning"])
            else:
                st.error(texts["health_error"])
        else:
            st.error(texts["health_error"])
    except:
        st.error(texts["connect_fail"])

st.markdown("""
<style>
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f0f2f6;
        color: black !important; /* Set text color to black */
        transition: color 0.3s ease; /* Smooth transition for hover effect */
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: red !important; /* Change text color to red on hover */
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important; /* Selected tab text color */
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: black; /* Button text color */
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)



st.markdown("---")
st.markdown("Powered by Gemini, FAISS, and LangChain | © 2025 ESG AI Made By Hiba & Azza")