import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
import seaborn as sns
import re # Importamos herramientas para buscar texto exacto

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Analista ONErpm AI", page_icon="🎹", layout="centered")
st.title("🎹 Chat con DSP Global")
st.caption("Modo: Análisis Robusto (Filtro Anti-Errores)")
st.markdown("---")

# --- 1. CONEXIÓN API ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Error: No se detectó la API Key.")
    st.stop()

# --- 2. AUTO-DETECTAR MODELO ---
@st.cache_resource
def get_best_model():
    try:
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            return 'models/gemini-1.5-flash'
        
        preferences = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        for pref in preferences:
            if pref in available_models: return pref
        return available_models[0] if available_models else 'models/gemini-1.5-flash'
    except:
        return 'models/gemini-1.5-flash'

valid_model_name = get_best_model()

# --- 3. CARGA Y LIMPIEZA DE DATOS ---
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        
        # Limpieza de espacios en texto (Vital para evitar "0 resultados")
        cols_texto = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre']
        for col in cols_texto:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Asegurar numéricos
        if 'Year' in df.columns: df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        if 'Month' in df.columns: df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
        if 'Release Date' in df.columns: df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Error leyendo los datos: {e}")
        return None

with st.spinner('Cargando datos...'):
    df = load_data()

# --- 4. FUNCIÓN EXTRAER CÓDIGO (NUEVO) ---
def limpiar_codigo(texto_respuesta):
    """
    Busca el código Python dentro de la respuesta de la IA, ignorando saludos.
    """
    # Patrón para buscar lo que está entre ```python y ```
    patron = r"```python(.*?)```"
    coincidencia = re.search(patron, texto_respuesta, re.DOTALL)
    
    if coincidencia:
        # Si encuentra el bloque, devuelve solo el contenido
        return coincidencia.group(1).strip()
    elif "```" in texto_respuesta:
        # Si hay bloques sin etiqueta python
        patron_generico = r"```(.*?)```"
        coincidencia_gen = re.search(patron_generico, texto_respuesta, re.DOTALL)
        if coincidencia_gen:
            return coincidencia_gen.group(1).strip()
    
    # Si no hay bloques, intentamos limpiar el texto crudo por si acaso
    texto_limpio = texto_respuesta.replace("```python", "").replace("```", "")
    return texto_limpio.strip()

# --- 5. CHAT ---
if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hola. Estoy listo para analizar tus destaques."})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ej: Comparativa Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja_loading = st.empty()
            caja_loading.markdown(f"🤖 *Procesando solicitud...*")

            try:
                info_columnas = df.dtypes.to_markdown()
                unique_dsps = list(df['DSP'].unique())[:10] # Muestra primeros 10 para no saturar

                prompt_maestro = f"""
                Actúa como Científico de Datos (Python).
                
                CONTEXTO:
                - DataFrame `df` cargado.
                - 1 FILA = 1 DESTAQUE (Placement).
                - Columnas clave: `DSP`, `Year`, `Month`, `Release Date`.
                - Valores DSP reales: {unique_dsps}...
                
                USUARIO: "{prompt}"
                
                INSTRUCCIONES:
                1. Genera código Python para Streamlit.
                2. Filtra estrictamente. Ej: df[(df['Year']==2025) & (df['Month']==1)].
                3. IMPRIME RESULTADOS INTERMEDIOS: Usa `st.write(f"Filas encontradas 2025: {{len(df_2025)}}")` para depurar.
                4. Usa `st.metric` para mostrar la variación.
                5. Genera gráficas con `fig, ax = plt.subplots()` y `st.pyplot(fig)`.
                
                IMPORTANTE: NO escribas texto fuera del bloque de código.
                """

                model = genai.GenerativeModel(valid_model_name)
                response = model.generate_content(prompt_maestro)
                
                # AQUI USAMOS EL NUEVO FILTRO
                codigo_seguro = limpiar_codigo(response.text)
                
                caja_loading.empty()
                
                # Ejecución
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt, "sns": sns}
                exec(codigo_seguro, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Resultado generado."})

            except Exception as e:
                caja_loading.error(f"Error técnico: {str(e)}")
                with st.expander("Ver código que falló"):
                    # Mostramos el código que intentó ejecutar para entender el error
                    if 'codigo_seguro' in locals():
                        st.code(codigo_seguro)
