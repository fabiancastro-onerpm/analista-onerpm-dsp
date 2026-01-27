import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Analista ONErpm AI", page_icon="🎹", layout="centered")
st.title("🎹 Chat con Datos ONErpm")
st.caption("Modo: Datos Reales (Sin Alucinaciones)")
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
        # Intentamos listar modelos, si falla, usamos el string directo
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            return 'models/gemini-1.5-flash' # Fallback seguro
            
        preferences = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        for pref in preferences:
            if pref in available_models: return pref
        return available_models[0] if available_models else 'models/gemini-1.5-flash'
    except:
        return 'models/gemini-1.5-flash'

valid_model_name = get_best_model()

# --- 3. CARGA Y LIMPIEZA DE DATOS (CRUCIAL) ---
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        # Leer datos
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        
        # --- LIMPIEZA PROFUNDA (PARA EVITAR ALUCINACIONES) ---
        
        # 1. Limpiar espacios en blanco en columnas de texto (Ej: "Spotify " -> "Spotify")
        cols_texto = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre']
        for col in cols_texto:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # 2. Asegurar que Año y Mes sean números
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        if 'Month' in df.columns:
            # Si el mes viene como texto ("Enero"), intentamos mapearlo o forzar numérico
            # Si ya son números, aseguramos que sean int
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce')

        # 3. Fechas
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Error leyendo los datos: {e}")
        return None

with st.spinner('Limpiando y organizando datos...'):
    df = load_data()

# --- 4. CHAT INTELIGENTE ---
if df is not None:
    # Verificador rápido para TI (Opcional, para ver qué detectó realmente)
    with st.expander("🔍 Ver datos que la IA está leyendo (Depuración)"):
        st.write("DSPs únicos encontrados:", df['DSP'].unique())
        st.write("Años únicos encontrados:", df['Year'].unique())
        st.write(f"Total filas: {len(df)}")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Listo. He limpiado los datos de espacios y errores. ¿Qué analizamos?"})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ej: Diferencia Spotify Enero 2025 vs 2026"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            caja_loading = st.empty()
            caja_loading.markdown(f"🕵️ *Investigando datos reales...*")

            try:
                # Metadata para que la IA entienda qué tiene
                info_columnas = df.dtypes.to_markdown()
                # Mostramos los valores únicos de DSP para que sepa cómo buscarlos
                unique_dsps = list(df['DSP'].unique())
                
                prompt_maestro = f"""
                Eres un Analista de Datos Senior (Python/Pandas).
                
                DATOS DISPONIBLES:
                - DataFrame `df` cargado.
                - Columnas: {info_columnas}
                - VALORES REALES EN COLUMNA 'DSP': {unique_dsps}
                
                INSTRUCCIONES DE VERDAD (NO ALUCINAR):
                1. El usuario pregunta: "{prompt}"
                2. CADA FILA ES UN DESTAQUE (Placement). Cuenta filas `len(df)`.
                3. FILTRADO:
                   - Usa `df['DSP'] == 'ValorExacto'` (copia del array de arriba).
                   - O usa `df['DSP'].str.contains('Spotify', case=False)` para ser seguro.
                   - Para FECHAS: Usa las columnas `Year` y `Month` (numéricos).
                4. ANTES DE CALCULAR PORCENTAJES:
                   - Muestra con `st.write` cuántos registros exactos encontraste para cada año.
                   - Ejemplo: "Encontré X filas para 2025 y Y filas para 2026".
                   - SI ALGUNO DA 0, AVISA AL USUARIO que no hay datos para ese filtro.
                5. Genera SOLO código Python ejecutable para Streamlit.
                """

                model = genai.GenerativeModel(valid_model_name)
                response = model.generate_content(prompt_maestro)
                codigo = response.text.replace("```python", "").replace("```", "").replace("plt.show()", "").strip()
                
                caja_loading.empty()
                local_vars = {"df": df, "pd": pd, "st": st, "plt": plt, "sns": sns}
                exec(codigo, {}, local_vars)
                st.session_state.messages.append({"role": "assistant", "content": "✅ Análisis completado."})

            except Exception as e:
                caja_loading.error(f"Error en el código generado: {str(e)}")
