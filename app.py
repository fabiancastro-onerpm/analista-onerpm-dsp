import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import plotly.graph_objects as go
import re
import time

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Analista ONErpm Pro", page_icon="🎹", layout="wide")

# Estilos CSS para mejorar la apariencia
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("❌ Falta API Key en Secrets.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. SELECTOR DE MODELO INTELIGENTE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🧠 Configuración")
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Preferencia: Pro -> Flash
        model_options = sorted(models, key=lambda x: 'flash' in x) 
        selected_model = st.selectbox("Modelo IA:", model_options, index=0)
        st.info("💡 Usa 'Flash' para velocidad. Usa 'Pro' para lógica compleja.")
    except:
        selected_model = "models/gemini-1.5-flash"

# -----------------------------------------------------------------------------
# 3. ETL MAESTRO (LIMPIEZA DE DATOS)
# -----------------------------------------------------------------------------
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_and_clean_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # Cargar DSP COPY
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        
        # --- A. LIMPIEZA DE COLUMNAS ---
        # Quitar espacios y saltos de línea en los nombres de las columnas
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # --- B. TEXTO A MAYÚSCULAS (NORMALIZACIÓN) ---
        cols_texto = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory', 'Origin']
        for col in cols_texto:
            if col in df.columns:
                # Se crea columna _CLEAN: " Spotify " -> "SPOTIFY"
                df[f"{col}_CLEAN"] = df[col].astype(str).fillna("UNKNOWN").str.strip().str.upper()

        # --- C. FECHAS Y NÚMEROS (LA PARTE CRÍTICA) ---
        
        # 1. AÑO (Year)
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

        # 2. MES (Month) - DICCIONARIO EXTENDIDO
        if 'Month' in df.columns:
            meses_map = {
                'ENERO':1, 'ENE':1, 'JANUARY':1, 'JAN':1, '01':1, '1':1,
                'FEBRERO':2, 'FEB':2, 'FEBRUARY':2, '02':2, '2':2,
                'MARZO':3, 'MAR':3, 'MARCH':3, '03':3, '3':3,
                'ABRIL':4, 'ABR':4, 'APRIL':4, 'APR':4, '04':4, '4':4,
                'MAYO':5, 'MAY':5, '05':5, '5':5,
                'JUNIO':6, 'JUN':6, 'JUNE':6, '06':6, '6':6,
                'JULIO':7, 'JUL':7, 'JULY':7, '07':7, '7':7,
                'AGOSTO':8, 'AGO':8, 'AUGUST':8, 'AUG':8, '08':8, '8':8,
                'SEPTIEMBRE':9, 'SEP':9, 'SEPTEMBER':9, '09':9, '9':9,
                'OCTUBRE':10, 'OCT':10, 'OCTOBER':10, '10':10,
                'NOVIEMBRE':11, 'NOV':11, 'NOVEMBER':11, '11':11,
                'DICIEMBRE':12, 'DIC':12, 'DECEMBER':12, 'DEC':12, '12':12
            }
            
            def normalizar_mes(val):
                if pd.isna(val): return 0
                if isinstance(val, (int, float)): return int(val)
                val_str = str(val).strip().upper()
                if val_str.isdigit(): return int(val_str)
                # Buscar en el mapa
                return meses_map.get(val_str, 0) # 0 si no encuentra nada

            df['Month_CLEAN'] = df['Month'].apply(normalizar_mes)
        else:
            df['Month_CLEAN'] = 0

        # --- D. FECHAS COMPLETAS ---
        # Buscamos columnas que parezcan fechas de inclusión
        cols_fecha = [c for c in df.columns if 'Inclusion' in c or 'Release' in c]
        for col in cols_fecha:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error crítico cargando datos: {e}")
        return None

df = load_and_clean_data()

# -----------------------------------------------------------------------------
# 4. INTERFAZ Y DEBUGGER VISUAL (PARA TI)
# -----------------------------------------------------------------------------
if df is not None:
    st.title("🎹 ONErpm Data Analyst (Gráficos Pro)")
    
    # --- BARRA LATERAL ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Auditoría de Datos")
        st.write(f"**Total Filas:** {len(df)}")
        
        # Conteo por Año (La prueba de fuego)
        if 'Year' in df.columns:
            st.write("📊 **Registros por Año:**")
            conteo_year = df['Year'].value_counts().sort_index()
            st.dataframe(conteo_year, height=150)
        
        # DSPs Detectados
        if 'DSP_CLEAN' in df.columns:
            with st.expander("Ver DSPs Detectados"):
                st.write(df['DSP_CLEAN'].unique())

    # --- VISOR DE DATOS LIMPIOS (Expandible) ---
    with st.expander("🕵️‍♀️ Ver tabla de datos limpia (Click para abrir)"):
        st.warning("Estos son los datos EXACTOS que la IA va a leer. Si aquí faltan datos, revisa el Excel.")
        st.dataframe(df.head(50))

# -----------------------------------------------------------------------------
# 5. CHATBOT CON GRÁFICOS INTERACTIVOS
# -----------------------------------------------------------------------------

def clean_code(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.replace("```python", "").replace("```", "").strip()

def generar_con_reintento(model, prompt):
    try:
        return model.generate_content(prompt)
    except Exception as e:
        if "429" in str(e):
            st.warning("⏳ Tráfico alto. Esperando 20s...")
            time.sleep(20)
            return model.generate_content(prompt)
        raise e

if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hola. Soy tu analista experto. Uso **Plotly** para gráficos y datos limpios."})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], str):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Diferencia porcentual Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info("🎨 Diseñando gráficos y calculando...")

            try:
                # Contexto
                dsps = list(df['DSP_CLEAN'].unique()) if 'DSP_CLEAN' in df.columns else []
                
                prompt_maestro = f"""
                Actúa como Data Analyst Senior experto en Visualización.
                
                DATOS DISPONIBLES (DataFrame `df`):
                - 1 Fila = 1 Destaque (Placement).
                - Columnas LIMPIAS: `DSP_CLEAN`, `Year` (int), `Month_CLEAN` (int).
                - DSPs reales: {dsps}
                
                REGLAS OBLIGATORIAS:
                1. **FILTRADO**:
                   - Usa `df['DSP_CLEAN'] == 'SPOTIFY'` (Mayúsculas).
                   - Usa `Year` y `Month_CLEAN`.
                
                2. **VISUALIZACIÓN (¡MUY IMPORTANTE!)**:
                   - NO uses matplotlib.
                   - USA SIEMPRE `plotly.express` (px) o `plotly.graph_objects` (go).
                   - Crea las figuras: `fig = px.bar(...)` o `fig = px.pie(...)`.
                   - Muestra la figura con `st.plotly_chart(fig, use_container_width=True)`.
                   - Agrega etiquetas de texto a las barras (`text_auto=True`).
                
                3. **LÓGICA DE NEGOCIO**:
                   - Calcula diferencias absolutas y porcentuales.
                   - Muestra métricas clave con `st.metric(label="...", value="...", delta="...")`.
                   - IMPRIME DEBUG: `st.write(f"Encontré {{len(df_filtrado)}} filas...")`.
                
                Genera SOLO código Python.
                """
                
                model = genai.GenerativeModel(selected_model)
                response = generar_con_reintento(model, prompt_maestro)
                code = clean_code(response.text)
                
                caja.empty()
                local_vars = {
                    "df": df, "pd": pd, "st": st, 
                    "px": px, "go": go, "time": time
                }
                exec(code, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis visual generado."})

            except Exception as e:
                caja.error(f"Error: {e}")
                with st.expander("Ver código generado"):
                    st.code(code)
