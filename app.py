import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="Analista ONErpm AI", page_icon="🎹", layout="centered")

st.title("🎹 Chat con Datos ONErpm")
st.markdown("Conectado a: **DSP Global - Pestaña: DSP COPY**")

# ---------------------------------------------------------
# 1. CONEXIÓN CON GEMINI (CEREBRO)
# ---------------------------------------------------------
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Falta configurar la GOOGLE_API_KEY en los Secrets de Streamlit.")
    st.stop()

# ---------------------------------------------------------
# 2. CONEXIÓN CON GOOGLE SHEETS (DATOS EN LA NUBE)
# ---------------------------------------------------------
# URL proporcionada del archivo
url_sheet = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600) # Se refresca cada 10 minutos automáticamente
def load_data():
    # Establecemos conexión usando las credenciales de secrets
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    try:
        # Leemos ESPECÍFICAMENTE la pestaña "DSP COPY"
        # usecols=None lee todas las columnas
        df = conn.read(spreadsheet=url_sheet, worksheet="DSP COPY")
        
        # Pequeña limpieza automática de fechas para evitar errores
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Error leyendo el Google Sheet. Verifica que la pestaña 'DSP COPY' exista y el robot tenga permisos.\nDetalle: {str(e)}")
        return None

with st.spinner('Descargando datos actualizados de la nube...'):
    df = load_data()

# ---------------------------------------------------------
# 3. INTERFAZ DE CHAT Y LÓGICA
# ---------------------------------------------------------

if df is not None:
    # Opcional: Mostrar tabla cruda para verificar
    with st.expander("🔍 Ver vista previa de los datos cargados"):
        st.dataframe(df.head(5))
        st.caption(f"Total de filas cargadas: {len(df)}")

    # Historial de chat en memoria
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Repintar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del Usuario
    prompt = st.chat_input("Ej: ¿Diferencia % de Spotify en Enero 2025 vs 2026?")

    if prompt:
        # 1. Mostrar pregunta usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generar respuesta asistente
        with st.chat_message("assistant"):
            caja_mensaje = st.empty()
            caja_mensaje.markdown("⏳ *Analizando datos...*")

            try:
                # --- PREPARAR EL PROMPT PARA GEMINI ---
                # Le damos metadata, no los datos completos (por privacidad y límites)
                info_columnas = df.dtypes.to_markdown()
                muestra_datos = df.head(3).to_markdown(index=False)

                prompt_maestro = f"""
                Actúa como un analista de datos Senior experto en Pandas (Python).
                Tienes disponible un DataFrame cargado en la variable `df`.
                
                METADATA DEL DATAFRAME:
                - Columnas y tipos: {info_columnas}
                - Ejemplo (primeras filas): {muestra_datos}
                
                PREGUNTA DEL USUARIO: "{prompt}"
                
                TU MISIÓN:
                Genera SOLAMENTE el bloque de código Python necesario para resolver la duda.
                
                REGLAS:
                1. Asume que `df` y `pd` (pandas) ya están importados.
                2. Guarda el resultado final (texto formateado, número o tabla markdown) en una variable llamada `resultado`.
                3. Si calculas porcentajes, formatéalos (ej: "23.5%").
                4. NO uses print().
                5. NO escribas ```python al inicio ni al final. Solo dame el código puro.
                """

                # Llamar a Gemini
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt_maestro)
                
                # Limpiar el código por si acaso
                codigo_generado = response.text.replace("```python", "").replace("```", "").strip()

                # --- EJECUCIÓN SEGURA DEL CÓDIGO ---
                variables_locales = {"df": df, "pd": pd}
                exec(codigo_generado, {}, variables_locales)
                
                # Obtener respuesta final de la variable 'resultado'
                respuesta_final = variables_locales.get("resultado", "No pude calcular una respuesta exacta.")

                # Mostrar y guardar
                caja_mensaje.markdown(respuesta_final)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
                
                # Opcional: Mostrar código usado en un expander
                with st.expander("Ver lógica aplicada (Código)"):
                    st.code(codigo_generado)

            except Exception as e:
                caja_mensaje.error(f"Ocurrió un error al procesar: {e}")
