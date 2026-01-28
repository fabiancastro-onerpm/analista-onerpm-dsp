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
st.set_page_config(page_title="Analista ONErpm - DSP COPY", page_icon="🎹", layout="wide")

# Conexión API
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("❌ Falta API Key en Secrets.")
    st.stop()

# Selector de Modelo (Sidebar)
with st.sidebar:
    st.header("🧠 Configuración")
    try:
        # Intentamos listar modelos disponibles
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Ponemos el PRO primero (mejor razonamiento)
        model_options = sorted(models, key=lambda x: 'pro' not in x)
        selected_model = st.selectbox("Modelo:", model_options, index=0)
    except:
        # Fallback si falla la lista
        selected_model = "models/gemini-1.5-flash"
        st.warning("Usando modelo Flash por defecto.")

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS (Pestaña: DSP COPY)
# -----------------------------------------------------------------------------
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_data_dsp_copy():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # LEER PESTAÑA EXACTA "DSP COPY"
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        
        # ---------------------------------------------------------
        # FASE DE LIMPIEZA TÉCNICA (ETL)
        # ---------------------------------------------------------
        
        # 1. Limpiar encabezados de columna (quitar \n y espacios extra)
        # Esto arregla "Inclusion Date \nMM/DD/YYYY" -> "Inclusion Date MM/DD/YYYY"
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # 2. Normalizar Texto (DSP, Artista, Territorio)
        # Creamos columnas "_CLEAN" para que los filtros sean a prueba de balas
        cols_texto = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Origin', 'Territory', 'Business Unit']
        for col in cols_texto:
            if col in df.columns:
                # Convertir a String -> Mayúsculas -> Quitar espacios
                df[f"{col}_CLEAN"] = df[col].astype(str).fillna("UNKNOWN").str.strip().str.upper()
        
        # 3. Normalizar Fechas y Números (Year, Month)
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            
        if 'Month' in df.columns:
            # Diccionario para meses en texto
            mapa_meses = {'ENERO':1, 'JANUARY':1, 'FEBRERO':2, 'FEBRUARY':2, 'MARZO':3, 'MARCH':3, 
                          'ABRIL':4, 'APRIL':4, 'MAYO':5, 'MAY':5, 'JUNIO':6, 'JUNE':6}
            
            def limpiar_mes(val):
                if isinstance(val, (int, float)): return val
                s = str(val).upper().strip()
                if s.isdigit(): return int(s)
                return mapa_meses.get(s, 0)

            df['Month'] = df['Month'].apply(limpiar_mes)
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(0).astype(int)

        # 4. Fechas Completas
        # Buscamos columnas de fecha típicas
        cols_fecha = ['Release Date', 'Inclusion Date', 'Inclusion Date MM/DD/YYYY']
        for col in cols_fecha:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error cargando la pestaña 'DSP COPY': {e}")
        return None

df = load_data_dsp_copy()

# -----------------------------------------------------------------------------
# 3. MONITOR DE DATOS (BARRA LATERAL)
# -----------------------------------------------------------------------------
if df is not None:
    st.title("🎹 Analista ONErpm (DSP COPY)")
    
    with st.sidebar:
        st.markdown("---")
        st.header("🔍 Auditoría de Datos")
        st.write(f"**Filas Totales:** {len(df)}")
        
        if 'Year' in df.columns:
            st.write(f"📅 **Años:** {sorted(df[df['Year']>0]['Year'].unique())}")
            
            # Chequeo rápido para ti
            c2025 = len(df[df['Year']==2025])
            c2026 = len(df[df['Year']==2026])
            st.caption(f"Registros 2025: {c2025}")
            st.caption(f"Registros 2026: {c2026}")
        
        if 'DSP_CLEAN' in df.columns:
            st.write("**DSPs Detectados:**")
            st.code(sorted(df['DSP_CLEAN'].unique()))

# -----------------------------------------------------------------------------
# 4. CHAT INTELIGENTE
# -----------------------------------------------------------------------------
def clean_code(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.replace("```python", "").replace("```", "").strip()

if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Conectado a **DSP COPY**. Datos limpios y listos."})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Diferencia Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info("🧠 Procesando...")

            try:
                # Contexto para el Prompt
                dsps = list(df['DSP_CLEAN'].unique()) if 'DSP_CLEAN' in df.columns else []
                
                prompt_maestro = f"""
                Eres un Data Analyst Senior.
                
                CONTEXTO:
                - Pestaña: "DSP COPY"
                - DataFrame: `df`
                - Columnas LIMPIAS: `DSP_CLEAN`, `Artist_CLEAN`, `Year` (int), `Month` (int).
                - DSPs Disponibles: {dsps}
                
                INSTRUCCIONES DE CÓDIGO:
                1. **FILTRADO:**
                   - Usa `df['DSP_CLEAN'] == 'SPOTIFY'` (Mayúsculas). NO uses la columna 'DSP' original.
                   - Usa `Year` y `Month` para fechas.
                
                2. **LÓGICA:**
                   - 1 Fila = 1 Placement. Usa `len(df)`.
                
                3. **DEBUG OBLIGATORIO:**
                   - Antes de mostrar el resultado final, IMPRIME cuántas filas encontraste.
                   - `st.write(f"Encontré {{len(df_2025)}} filas en 2025")`.
                
                4. **SALIDA:** Solo código Python.
                
                Usuario: "{prompt}"
                """
                
                model = genai.GenerativeModel(selected_model)
                response = model.generate_content(prompt_maestro)
                code = clean_code(response.text)
                
                caja.empty()
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt, "sns": sns}
                exec(code, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis completado."})

            except Exception as e:
                caja.error(f"Error: {e}")
