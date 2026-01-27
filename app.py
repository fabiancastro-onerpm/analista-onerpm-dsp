import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y DIAGNÓSTICO VISUAL
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Analista ONErpm (Modo Preciso)", page_icon="🎹", layout="wide")

with st.sidebar:
    st.header("🔧 Panel de Diagnóstico")
    st.info("Aquí verás qué datos reales está leyendo el sistema.")

# --- CONEXIÓN API ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("❌ FALTA API KEY: Ve a Settings -> Secrets y configurala.")
    st.stop()

# --- DETECTOR DE MODELO ---
@st.cache_resource
def get_model_name():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Preferimos Flash por velocidad, luego Pro
        for pref in ['models/gemini-1.5-flash', 'models/gemini-pro']:
            if pref in models: return pref
        return models[0] if models else 'models/gemini-1.5-flash'
    except:
        return 'models/gemini-1.5-flash'

MODEL_NAME = get_model_name()
st.sidebar.success(f"🤖 Cerebro activo: {MODEL_NAME.split('/')[-1]}")

# -----------------------------------------------------------------------------
# 2. CARGA Y LIMPIEZA DE DATOS (LA PARTE MÁS IMPORTANTE)
# -----------------------------------------------------------------------------
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_and_clean_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        
        # --- LIMPIEZA AGRESIVA ---
        # 1. Convertir nombres de columnas a limpio (sin espacios extra)
        df.columns = df.columns.str.strip()
        
        # 2. Convertir columnas de TEXTO clave a Mayúsculas y sin espacios (Para búsquedas infalibles)
        # Creamos columnas "NORMALIZADAS" internas para buscar
        if 'DSP' in df.columns:
            df['DSP_NORM'] = df['DSP'].astype(str).str.upper().str.strip()
        
        # 3. Forzar AÑO y MES a números enteros
        cols_num = ['Year', 'Month', 'Week', 'Q']
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # 4. Fechas
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None

df = load_and_clean_data()

# -----------------------------------------------------------------------------
# 3. VERIFICACIÓN DE DATOS (PARA QUE NO TE MIENTA)
# -----------------------------------------------------------------------------
if df is not None:
    # Mostramos en la barra lateral lo que REALMENTE hay
    with st.sidebar:
        st.markdown("---")
        st.write(f"📊 **Filas Totales:** {len(df)}")
        
        if 'Year' in df.columns:
            years = sorted(df['Year'].unique())
            st.write(f"📅 **Años detectados:** {years}")
            
        if 'DSP' in df.columns:
            dsps = df['DSP'].unique()
            st.write(f"🎧 **DSPs detectados ({len(dsps)}):**")
            st.code(dsps)

    # Título principal
    st.title("🎹 Chat de Datos (Sin Alucinaciones)")
    st.markdown("""
    Este modo muestra los pasos intermedios. Si dice "0 filas encontradas", 
    revisa el Panel de Diagnóstico a la izquierda para ver si el año existe.
    """)

# -----------------------------------------------------------------------------
# 4. LÓGICA DEL CHAT
# -----------------------------------------------------------------------------
def extract_code(text):
    """Limpia el texto para sacar solo el código Python"""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.replace("```python", "").replace("```", "").strip()

if df is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: Diferencia Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja = st.empty()
            caja.info("🕵️ Validando datos y generando código...")

            try:
                # PREPARAMOS EL CONTEXTO PERFECTO
                columnas = list(df.columns)
                # Le damos los valores UNICOS REALES para que no adivine
                unique_dsp_list = list(df['DSP_NORM'].unique()) if 'DSP_NORM' in df.columns else []
                unique_years = list(df['Year'].unique()) if 'Year' in df.columns else []
                
                prompt_maestro = f"""
                Eres un Experto Data Scientist en Python.
                
                TU OBJETIVO: Generar código Python para responder: "{prompt}"
                
                TIENES ESTOS DATOS REALES (NO INVENTES OTROS):
                - DataFrame: `df`
                - Columnas: {columnas}
                - Años disponibles (int): {unique_years}
                - DSPs disponibles (NORMALIZADOS MAYÚSCULAS): {unique_dsp_list}
                
                REGLAS ESTRICTAS DE FILTRADO:
                1. PARA FILTRAR TEXTO (Artist, DSP, etc):
                   - Usa SIEMPRE `.str.upper().str.strip()` o la columna `DSP_NORM`.
                   - Ejemplo CORRECTO: `df[df['DSP_NORM'] == 'SPOTIFY']`
                   - Ejemplo INCORRECTO: `df[df['DSP'] == 'Spotify']` (Esto falla por mayúsculas).
                
                2. PARA FILTRAR FECHAS:
                   - Usa las columnas numéricas `Year` y `Month` siempre que sea posible.
                   - Ejemplo: `df[(df['Year'] == 2025) & (df['Month'] == 1)]`
                
                3. REGLA "CHISMOSA" (DEBUG):
                   - ANTES de dar el resultado final, debes imprimir cuántas filas encontraste en cada paso.
                   - Usa: `st.write(f"Paso 1: Encontré {{len(filtro1)}} filas para 2025")`
                   - Usa: `st.write(f"Paso 2: Encontré {{len(filtro2)}} filas para 2026")`
                   - Si len es 0, usa `st.error("No hay datos para este filtro")`.
                
                4. SALIDA:
                   - Tablas: `st.dataframe()`
                   - Texto: `st.write()`
                   - Gráficos: `fig, ax = plt.subplots()... st.pyplot(fig)`
                
                Genera SOLO el código Python.
                """

                model = genai.GenerativeModel(MODEL_NAME)
                response = model.generate_content(prompt_maestro)
                code = extract_code(response.text)
                
                caja.empty()
                
                # Entorno de ejecución seguro
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt, "sns": sns}
                exec(code, {}, local_vars)
                
                st.session_state.messages.append({"role": "assistant", "content": "✅ Ejecución finalizada."})

            except Exception as e:
                caja.error(f"Error técnico: {e}")
                with st.expander("Ver código generado (Debug)"):
                    st.code(code)
