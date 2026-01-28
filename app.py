import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN E INTERFAZ
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Analista ONErpm - AndReg&Car", page_icon="🎹", layout="wide")

# Conexión API
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("❌ Falta API Key en Secrets.")
    st.stop()

# Selector de Modelo (Sidebar)
with st.sidebar:
    st.header("🧠 Cerebro AI")
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Preferimos el PRO para razonamiento lógico
        model_options = sorted(models, key=lambda x: 'pro' not in x)
        selected_model = st.selectbox("Modelo:", model_options, index=0)
    except:
        selected_model = "models/gemini-1.5-flash"

# -----------------------------------------------------------------------------
# 2. CARGA Y LIMPIEZA DE DATOS (ETL DURO)
# -----------------------------------------------------------------------------
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_data_andreg():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # 1. LEER SOLO LA PESTAÑA CORRECTA
        df = conn.read(spreadsheet=url_sheet, worksheet="AndReg&Car")
        
        # ---------------------------------------------------------
        # FASE DE LIMPIEZA PROFUNDA (PRE-IA)
        # ---------------------------------------------------------
        
        # A. Limpiar Nombres de Columnas (Quitar saltos de línea y espacios)
        # Tu columna "Inclusion Date \nMM/DD/YYYY" se limpiará a "Inclusion Date MM/DD/YYYY"
        df.columns = df.columns.str.replace('\n', ' ').str.strip()
        
        # B. Estandarización de Texto (DSP, Artist, Playlist)
        # Creamos columnas _CLEAN para que el filtro sea insensible a mayúsculas/espacios
        cols_texto = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Origin']
        for col in cols_texto:
            if col in df.columns:
                # Convertir a String -> Mayúsculas -> Quitar espacios -> Rellenar vacíos
                df[f"{col}_CLEAN"] = df[col].astype(str).fillna("").str.strip().str.upper()
        
        # C. Estandarización de Fechas y Números (Year, Month)
        # Forzamos a que sean números enteros. Si hay error, pone 0.
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            
        if 'Month' in df.columns:
            # Diccionario por si el mes viene en texto
            mapa_meses = {'ENERO':1, 'JANUARY':1, 'FEBRERO':2, 'FEBRUARY':2, 'MARZO':3, 'MARCH':3, 
                          'ABRIL':4, 'APRIL':4, 'MAYO':5, 'MAY':5, 'JUNIO':6, 'JUNE':6} 
            
            # Función auxiliar para limpiar el mes
            def limpiar_mes(val):
                if isinstance(val, (int, float)): return val
                val_str = str(val).upper().strip()
                if val_str.isdigit(): return int(val_str)
                return mapa_meses.get(val_str, 0) # Devuelve 0 si no entiende

            df['Month'] = df['Month'].apply(limpiar_mes)
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int)

        # D. Release Date
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error cargando AndReg&Car: {e}")
        return None

df = load_data_andreg()

# -----------------------------------------------------------------------------
# 3. MONITOR DE VERDAD (SIDEBAR DEBUG)
# -----------------------------------------------------------------------------
if df is not None:
    st.title("🎹 Analista AndReg&Car (Blindado)")
    
    with st.sidebar:
        st.markdown("---")
        st.header("🔍 Datos Reales Detectados")
        st.caption("Esto es lo que Python ve antes de la IA:")
        
        st.metric("Total Filas (Destaques)", len(df))
        
        if 'Year' in df.columns:
            years_found = sorted(df[df['Year'] > 0]['Year'].unique())
            st.write(f"📅 **Años:** {years_found}")
            
            # --- DEBUGGER ESPECÍFICO 2025 vs 2026 ---
            c_2025 = len(df[df['Year'] == 2025])
            c_2026 = len(df[df['Year'] == 2026])
            st.write(f"👉 Filas Año 2025: **{c_2025}**")
            st.write(f"👉 Filas Año 2026: **{c_2026}**")

        if 'DSP_CLEAN' in df.columns:
            st.write("🎧 **DSPs (Normalizados):**")
            st.code(sorted(df['DSP_CLEAN'].unique()))

# -----------------------------------------------------------------------------
# 4. CHAT LOGIC (USANDO COLUMNAS LIMPIAS)
# -----------------------------------------------------------------------------
def clean_code(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.replace("```python", "").replace("```", "").strip()

if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hola. Analizo exclusivamente la pestaña **AndReg&Car**. Ya limpié los datos. ¿Qué necesitas?"})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Diferencia Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info("🧠 Generando análisis sobre datos limpios...")

            try:
                # Preparamos la lista de valores válidos para el prompt
                dsps_validos = list(df['DSP_CLEAN'].unique())
                
                prompt_maestro = f"""
                Eres un Data Analyst Python experto en ONErpm.
                
                CONTEXTO:
                - Estás analizando la pestaña "AndReg&Car".
                - Los datos YA HAN SIDO LIMPIADOS por Python.
                - DataFrame disponible: `df`
                
                COLUMNAS DISPONIBLES (USAR ESTAS):
                - Para filtrar DSP usa: `DSP_CLEAN` (Valores: {dsps_validos})
                - Para filtrar Artista usa: `Artist_CLEAN`
                - Fechas: `Year` (int), `Month` (int), `Release Date` (datetime).
                
                REGLAS OBLIGATORIAS:
                1. **FILTRADO:** Usa SIEMPRE las columnas `_CLEAN` y valores en MAYÚSCULAS.
                   - CORRECTO: `df[df['DSP_CLEAN'] == 'SPOTIFY']`
                   - INCORRECTO: `df[df['DSP'] == 'Spotify']`
                
                2. **LÓGICA:** - Cada fila es un destaque. Usa `len(df)` para contar.
                   - Si piden comparar Enero 2025 vs 2026, filtra por `Month==1` y `Year==2025`/`2026`.
                
                3. **VERIFICACIÓN (DEBUG):**
                   - IMPRIME SIEMPRE los conteos intermedios con `st.write()`.
                   - Ej: `st.write(f"Encontré {{len(df_2025)}} filas para 2025")`.
                
                4. **SALIDA:** Genera SOLO código Python.
                
                Usuario pregunta: "{prompt}"
                """
                
                model = genai.GenerativeModel(selected_model)
                response = model.generate_content(prompt_maestro)
                code = clean_code(response.text)
                
                caja.empty()
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt, "sns": sns}
                exec(code, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis completado."})

            except Exception as e:
                caja.error(f"Error de ejecución: {e}")
