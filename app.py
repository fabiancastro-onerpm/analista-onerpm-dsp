import streamlit as st
import pandas as pd
import google.generativeai as genai
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import plotly.graph_objects as go
import time
import re
import io

# ==============================================================================
# 1. CONFIGURACIÓN DEL SISTEMA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="ONErpm Data Analyst Enterprise",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS Profesionales
st.markdown("""
<style>
    /* Estilo de Métricas */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        padding: 15px;
        border-radius: 10px;
    }
    /* Estilo del Chat */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
    }
    /* Títulos */
    h1 {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Verificación de Seguridad
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("🚨 CRÍTICO: No se encontró la API Key en los Secrets. El sistema no puede iniciar.")
    st.stop()
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ==============================================================================
# 2. MOTOR ETL (EXTRACCIÓN, TRANSFORMACIÓN Y CARGA)
# ==============================================================================
URL_SHEET = "https://docs.google.com/spreadsheets/d/10y2YowTEgQYdWxs6c8D0fgJDDwGIT8_wyH0rQbERgG0/edit?gid=1919114384#gid=1919114384"

@st.cache_data(ttl=600, show_spinner=False)
def cargar_datos_maestros():
    """
    Función ETL Blindada:
    1. Conecta a Google Sheets.
    2. Limpia nombres de columnas (elimina caracteres ocultos).
    3. Normaliza texto (DSP, Artistas).
    4. Estandariza Fechas (Mapeo profundo de meses).
    """
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    try:
        # FASE 1: LECTURA
        df = conn.read(spreadsheet=URL_SHEET, worksheet="DSP COPY")
        
        # FASE 2: LIMPIEZA ESTRUCTURAL
        # Elimina saltos de línea y espacios en encabezados
        df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
        
        # FASE 3: NORMALIZACIÓN DE TEXTO (Para filtros infalibles)
        # Creamos columnas "shadow" (ocultas) _CLEAN para que la IA las use
        cols_texto = ['DSP', 'Artist', 'Title', 'Playlist', 'Genre', 'Territory', 'Origin', 'Business Unit']
        for col in cols_texto:
            if col in df.columns:
                df[f"{col}_CLEAN"] = df[col].astype(str).fillna("UNKNOWN").str.strip().str.upper()

        # FASE 4: INGENIERÍA DE FECHAS
        # A. Año
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

        # B. Mes (Diccionario Exhaustivo Multilenguaje)
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
            
            def normalizar_mes_robusto(val):
                if pd.isna(val): return 0
                if isinstance(val, (int, float)): return int(val)
                s = str(val).strip().upper()
                if s.isdigit(): return int(s)
                return meses_map.get(s, 0) # Retorna 0 si no lo entiende

            df['Month_CLEAN'] = df['Month'].apply(normalizar_mes_robusto)
        else:
            df['Month_CLEAN'] = 0 # Fallback si no existe columna

        # C. Fechas Datetime Reales
        cols_fecha = [c for c in df.columns if 'Inclusion' in c or 'Release' in c]
        for col in cols_fecha:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    except Exception as e:
        st.error(f"❌ Error Fatal en ETL: {str(e)}")
        return None

# Carga inicial con spinner visual
with st.spinner('🔄 Iniciando Motor de Datos ONErpm... Conectando a DSP COPY...'):
    df = cargar_datos_maestros()

# ==============================================================================
# 3. BARRA LATERAL (CONTROLES)
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png", width=50)
    st.title("Panel de Control")
    
    st.markdown("### 🧠 Configuración IA")
    try:
        # Lógica para priorizar Flash (Velocidad) pero permitir Pro (Complejidad)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Ordenamos: Flash primero
        model_options = sorted(available_models, key=lambda x: 'flash' in x, reverse=True)
        selected_model = st.selectbox("Modelo del Cerebro:", model_options, index=0, help="Flash es rápido. Pro es más inteligente.")
    except:
        selected_model = "models/gemini-1.5-flash"
        st.warning("⚠️ Modo Offline/Default activado")

    st.markdown("---")
    st.markdown("### 🛠️ Herramientas")
    if st.button("🗑️ Borrar Historial de Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Recargar Datos (Cache)"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    if df is not None:
        st.success(f"🟢 Sistema Online\nFilas cargadas: {len(df)}")

# ==============================================================================
# 4. INTERFAZ PRINCIPAL (PESTAÑAS)
# ==============================================================================

if df is not None:
    # Creamos pestañas para organizar la "magnitud" de la herramienta
    tab_chat, tab_data, tab_debug = st.tabs(["💬 Chat Analista", "📊 Inspector de Datos", "⚙️ Logs del Sistema"])

    # --------------------------------------------------------------------------
    # PESTAÑA 1: CHAT ANALISTA (La Core Feature)
    # --------------------------------------------------------------------------
    with tab_chat:
        st.header("🎹 Asistente de Inteligencia de Negocios")
        st.caption(f"Analizando datos normalizados con el modelo: **{selected_model}**")
        
        # Historial de Chat
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hola. Tengo acceso total a la base de datos limpia. Puedo generar gráficas comparativas, tablas dinámicas y análisis de tendencias. ¿Por dónde empezamos?"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if isinstance(msg["content"], str):
                    st.markdown(msg["content"])

        # Input del Usuario
        if prompt := st.chat_input("Ej: Comparativa de Destaques Spotify Enero 2025 vs 2026"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                status_box = st.status("🧠 Procesando solicitud...", expanded=True)
                
                # --- PREPARACIÓN DEL PROMPT DE ALTA INGENIERÍA ---
                dsps_list = list(df['DSP_CLEAN'].unique()) if 'DSP_CLEAN' in df.columns else []
                
                prompt_sistema = f"""
                Eres el Data Scientist Principal de ONErpm. Tu trabajo es extraer insights precisos.
                
                CONTEXTO DE DATOS (DataFrame `df`):
                - Total Filas: {len(df)}
                - 1 Fila = 1 Destaque (Placement) conseguido.
                - Columnas LIMPIAS (Úsalas para filtrar): `DSP_CLEAN`, `Artist_CLEAN`, `Year` (int), `Month_CLEAN` (int).
                - Lista de DSPs válidos: {dsps_list}
                
                REGLAS DE ORO (STRICT MODE):
                1. **FILTRADO INFALIBLE**:
                   - Para DSPs, usa SIEMPRE: `df['DSP_CLEAN'] == 'VALOR_EN_MAYUSCULAS'`.
                   - Ejemplo: `df[df['DSP_CLEAN'] == 'SPOTIFY']`. NUNCA uses la columna 'DSP' original.
                   - Para fechas, usa `Year` y `Month_CLEAN`.
                
                2. **VISUALIZACIÓN AVANZADA**:
                   - Usa la librería `plotly` (px o go). Es interactiva y profesional.
                   - Muestra gráficos con `st.plotly_chart(fig, use_container_width=True)`.
                   - Si haces gráficos de barras, agrega `text_auto=True` para ver los números.
                
                3. **COMUNICACIÓN DE RESULTADOS**:
                   - Usa `st.metric()` para los números grandes (KPIs).
                   - Si comparas fechas, calcula la variación porcentual.
                   - IMPORTANTE: Imprime mensajes de validación con `st.write(f"🔍 Encontré {{len(df_filtrado)}} registros...")`.
                
                4. **CÓDIGO**:
                   - Genera ÚNICA y EXCLUSIVAMENTE bloque de código Python ejecutable.
                   - No escribas explicaciones fuera del bloque de código.
                
                Pregunta del Usuario: "{prompt}"
                """

                # --- LÓGICA DE REINTENTO (ANTÍDOTO ERROR 429) ---
                def llamar_ia_con_reintento():
                    intentos = 0
                    while intentos < 3:
                        try:
                            status_box.write(f"📡 Conectando con Google Gemini (Intento {intentos+1})...")
                            model = genai.GenerativeModel(selected_model)
                            return model.generate_content(prompt_sistema)
                        except Exception as e:
                            if "429" in str(e):
                                status_box.warning(f"🚦 Tráfico alto detectado. Pausando 15s...")
                                time.sleep(15)
                                intentos += 1
                            else:
                                raise e
                    raise Exception("Tiempo de espera agotado. Intenta con el modelo Flash.")

                try:
                    # 1. Llamada a la IA
                    response = llamar_ia_con_reintento()
                    
                    # 2. Limpieza de Código (Regex)
                    texto_respuesta = response.text
                    match = re.search(r"```python(.*?)```", texto_respuesta, re.DOTALL)
                    codigo_final = match.group(1).strip() if match else texto_respuesta.replace("```python", "").replace("```", "").strip()
                    
                    status_box.write("⚙️ Ejecutando análisis en Python...")
                    
                    # 3. Ejecución Segura
                    # Pasamos todas las librerías necesarias al entorno local
                    variables_entorno = {
                        "df": df, "pd": pd, "st": st, "px": px, "go": go, "time": time, "re": re
                    }
                    exec(codigo_final, {}, variables_entorno)
                    
                    status_box.update(label="✅ Análisis Completado Exitosamente", state="complete", expanded=False)
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Aquí tienes el análisis."})

                except Exception as e:
                    status_box.update(label="❌ Error en el proceso", state="error")
                    st.error(f"Ocurrió un error técnico: {str(e)}")
                    with st.expander("Ver detalle técnico (Para desarrolladores)"):
                        st.code(str(e))
                        if 'codigo_final' in locals():
                            st.write("Código generado que falló:")
                            st.code(codigo_final)

    # --------------------------------------------------------------------------
    # PESTAÑA 2: INSPECTOR DE DATOS (Para que tú valides la "Verdad")
    # --------------------------------------------------------------------------
    with tab_data:
        st.header("🔎 Inspector de Datos Maestros")
        st.markdown("Utiliza esta sección para verificar qué datos está viendo realmente el sistema.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Destaques", len(df))
        with col2:
            st.metric("Columnas Totales", len(df.columns))
        with col3:
            registros_2025 = len(df[df['Year'] == 2025]) if 'Year' in df.columns else 0
            st.metric("Registros 2025", registros_2025)

        # Filtros rápidos
        st.markdown("#### Vista Previa de Datos Limpios")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Botón de Descarga (Feature Enterprise)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Data Limpia (CSV)",
            data=csv,
            file_name='onerpm_data_clean.csv',
            mime='text/csv',
        )

    # --------------------------------------------------------------------------
    # PESTAÑA 3: LOGS Y DIAGNÓSTICO (Para entender qué pasa por detrás)
    # --------------------------------------------------------------------------
    with tab_debug:
        st.header("⚙️ Logs del Sistema")
        st.info("Información técnica sobre el procesamiento de datos.")
        
        st.write("### 1. Mapa de Columnas Detectadas")
        st.json(list(df.columns))
        
        st.write("### 2. Valores Únicos de DSP (Normalizados)")
        if 'DSP_CLEAN' in df.columns:
            st.code(list(df['DSP_CLEAN'].unique()))
        else:
            st.error("No se detectó la columna DSP_CLEAN")

        st.write("### 3. Distribución Temporal")
        if 'Year' in df.columns and 'Month_CLEAN' in df.columns:
            conteo = df.groupby(['Year', 'Month_CLEAN']).size().reset_index(name='Counts')
            st.dataframe(conteo, use_container_width=True)

else:
    # Pantalla de Error Fatal si no hay datos
    st.error("❌ No se pudieron cargar los datos. Revisa la conexión con Google Sheets.")
