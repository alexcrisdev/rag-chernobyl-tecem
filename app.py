import streamlit as st
import pymongo
from google import genai
from google.genai import types
import numpy as np
import time

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan las variables de entorno GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

# =======================
# CLIENTES (cacheados)
# =======================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["pdf_embeddings_db"]
    return db["chernobyl_vectors"]

client_genai = get_genai_client()
collection = get_mongo_collection()

# =======================
# FUNCIONES
# =======================

def crear_embedding(texto: str):
    """
    Genera embedding de la query con el mismo modelo y dimensión que se usó
    al indexar (gemini-embedding-001, 3072 dims, normalizado L2).

    IMPORTANTE: para queries de búsqueda usar task_type='RETRIEVAL_QUERY'
    (al indexar se usó 'RETRIEVAL_DOCUMENT').
    """
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return response.embeddings[0].values

def buscar_similares(embedding, k=5):
    """
    Busca los documentos más similares en MongoDB Atlas Vector Search.
    Requiere el índice 'vector_index' creado sobre el campo 'embedding'.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))

MODELOS = ["gemini-3.8-flash", "gemini-3.5-flash"]  # principal y respaldo

def generar_respuesta(pregunta: str, contextos: list[dict]) -> str:
    """Usa Gemini para responder con contexto recuperado (RAG)."""
    contexto = "\n\n".join([c["texto"] for c in contextos])
    prompt = f"""Eres un asistente especializado en el informe "Chernobyl's Legacy" del Foro de Chernóbil (OIEA, OMS y otros organismos de la ONU, 2006).

Reglas:
- Usa EXCLUSIVAMENTE el contexto para responder. No agregues conocimiento externo.
- Si la respuesta no está en el contexto, responde: "El informe no contiene esa información."
- Conserva cifras, unidades y fechas exactamente como aparecen.
- Distingue los datos confirmados de las estimaciones o proyecciones.
- El contexto está en inglés: responde siempre en español, de forma concisa y clara.

Contexto:
{contexto}

Pregunta: {pregunta}"""

    ultimo_error = None
    for modelo in MODELOS:
        for intento in range(3):  # hasta 3 intentos por modelo
            try:
                response = client_genai.models.generate_content(
                    model=modelo,
                    contents=prompt,
                )
                return response.text + f"\n\n_Modelo: {modelo}_"
            except Exception as e:
                ultimo_error = e
                if getattr(e, "code", None) in (500, 503):  # saturado: esperar y reintentar
                    time.sleep(2 ** intento)                # 1 s, 2 s, 4 s
                else:
                    break                                   # otro error: pasar al respaldo
    raise ultimo_error

# =======================
# INTERFAZ STREAMLIT
# =======================

st.set_page_config(page_title="Chatbot Chernóbil | RAG", page_icon="☢️")
st.title("☢️ Chatbot sobre el legado de Chernóbil")
st.caption("Responde solo con el informe Chernobyl's Legacy del Foro de Chernóbil (OIEA/OMS, 2006).")

k = st.sidebar.slider("Fragmentos a recuperar (k)", min_value=1, max_value=10, value=5)

if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar historial PRIMERO (antes de procesar la nueva pregunta)
for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])

pregunta = st.chat_input("Pregunta sobre el accidente, la salud o el medio ambiente...")

if pregunta:
    # Mostrar inmediatamente la pregunta del usuario
    st.chat_message("user").write(pregunta)
    st.session_state.historial.append({"rol": "usuario", "texto": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Buscando respuesta..."):
            try:
                emb = crear_embedding(pregunta)
                similares = buscar_similares(emb, k=k)

                if not similares:
                    respuesta = "No encontré información relevante en el documento."
                else:
                    respuesta = generar_respuesta(pregunta, similares)
            except Exception as e:
                respuesta = f"⚠️ Ocurrió un error: {e}"

        st.write(respuesta)

        # Opcional: mostrar fuentes recuperadas
        if 'similares' in locals() and similares:
            with st.expander("🔍 Fragmentos recuperados"):
                for i, c in enumerate(similares, 1):
                    st.markdown(f"**Fragmento {i}** — score: `{c['score']:.4f}`")
                    st.write(c["texto"][:500] + ("…" if len(c["texto"]) > 500 else ""))
                    st.divider()

    st.session_state.historial.append({"rol": "bot", "texto": respuesta})
