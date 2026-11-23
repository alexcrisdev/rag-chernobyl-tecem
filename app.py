import streamlit as st
import pymongo
from google import genai
from google.genai import types
import numpy as np
import re
import time

# =======================
# CONFIGURACIÓN
# =======================

TIMEOUT_MS = 30000        # tiempo máximo por llamada a Gemini
LIMITE_TOTAL_S = 90       # tiempo máximo probando modelos
MAX_MODELOS = 5           # cuántos modelos probar como máximo
MODELOS_PREFERIDOS = ["gemini-3.8-flash", "gemini-3.5-flash"]
EXCLUIR = ("image", "tts", "audio", "live", "robotics", "computer", "embedding", "native")

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
    return genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options=types.HttpOptions(timeout=TIMEOUT_MS),
    )

@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
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
    for intento in range(3):
        try:
            response = client_genai.models.embed_content(
                model="gemini-embedding-001",
                contents=texto,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                ),
            )
            return response.embeddings[0].values
        except Exception as e:
            if getattr(e, "code", None) in (400, 401, 403) or intento == 2:
                raise
            time.sleep(2 ** intento)

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

def orden_modelo(nombre: str):
    """Primero los estables y, dentro de ellos, las versiones más nuevas."""
    version = re.search(r"gemini-(\d+(?:\.\d+)?)", nombre)
    numero = float(version.group(1)) if version else 0.0
    return ("preview" in nombre or "exp" in nombre, "lite" in nombre, -numero, nombre)

@st.cache_data(ttl=3600, show_spinner=False)
def listar_modelos() -> list[str]:
    """Busca los modelos Flash de texto disponibles para esta API key."""
    disponibles = []
    for m in client_genai.models.list():
        nombre = m.name.removeprefix("models/")
        acciones = m.supported_actions or []
        if ("generateContent" in acciones and "flash" in nombre
                and not any(x in nombre for x in EXCLUIR)):
            disponibles.append(nombre)
    preferidos = [m for m in MODELOS_PREFERIDOS if m in disponibles]
    otros = sorted((m for m in disponibles if m not in preferidos), key=orden_modelo)
    return (preferidos + otros)[:MAX_MODELOS]

def modelos_candidatos() -> list[str]:
    try:
        return listar_modelos() or MODELOS_PREFERIDOS
    except Exception:
        return MODELOS_PREFERIDOS

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

    inicio = time.monotonic()
    ultimo_error = None
    for modelo in modelos_candidatos():
        for intento in range(2):  # saturado: un reintento y luego otro modelo
            if time.monotonic() - inicio > LIMITE_TOTAL_S:
                raise TimeoutError("Gemini no respondió a tiempo. Intenta en unos minutos.")
            try:
                response = client_genai.models.generate_content(
                    model=modelo,
                    contents=prompt,
                )
                if not response.text:
                    raise ValueError(f"{modelo} devolvió una respuesta vacía")
                return response.text + f"\n\n_Modelo: {modelo}_"
            except Exception as e:
                ultimo_error = e
                codigo = getattr(e, "code", None)
                if codigo in (401, 403):      # clave inválida: otro modelo no ayuda
                    raise
                if codigo in (500, 503) and intento == 0:
                    time.sleep(1)
                    continue
                break                         # timeout, cuota u otro error: siguiente modelo
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
