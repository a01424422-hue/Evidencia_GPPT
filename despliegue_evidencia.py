
#####################################################
# Importamos librerías
#####################################################
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import ast

#####################################################
# Funciones Helper Generales
#####################################################

def get_numeric_columns(df: pd.DataFrame) -> list:
    """Devuelve las columnas numéricas puras (int/float)."""
    return df.select_dtypes(include=["int", "float", "int64", "float64"]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame, max_unique_for_numeric_as_cat: int = 10) -> list:
    """Devuelve columnas categóricas y numéricas de baja cardinalidad."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in get_numeric_columns(df):
        if col in df.columns and df[col].nunique(dropna=True) <= max_unique_for_numeric_as_cat:
            if col not in cat_cols:
                cat_cols.append(col)
    return cat_cols

@st.cache_resource
def load_data(path: str) -> pd.DataFrame:
    """Carga CSV, elimina 'Unnamed: 0' y añade 'origen_archivo'."""
    try:
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df['origen_archivo'] = path.split('/')[-1]
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo {path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al leer {path}: {e}")
        return pd.DataFrame()

#####################################################
# Funciones de Limpieza y Pre-procesamiento
#####################################################

def preprocesar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte 'price' a float numérico y luego a USD."""
    df_proc = df.copy()
    EUR_TO_USD = 1.10; MXN_TO_USD = 0.055

    if "price" in df_proc.columns:
        if not pd.api.types.is_numeric_dtype(df_proc['price']):
            df_proc["price"] = pd.to_numeric(
                df_proc["price"].astype(str).replace(r"[\\$,]", "", regex=True),
                errors='coerce'
            )
        else: df_proc["price"] = df_proc["price"].astype(float)

        if 'origen_archivo' in df_proc.columns:
            mask_venecia = df_proc['origen_archivo'] == 'Venecia.csv'
            if mask_venecia.any(): df_proc.loc[mask_venecia, 'price'] *= EUR_TO_USD
            mask_cdmx = df_proc['origen_archivo'] == 'CDMX.csv'
            if mask_cdmx.any(): df_proc.loc[mask_cdmx, 'price'] *= MXN_TO_USD
            df_proc.rename(columns={'price': 'price_usd'}, inplace=True)
        else:
            st.warning("Columna 'origen_archivo' no encontrada, no se convirtió moneda.")
            if "price" in df_proc.columns: df_proc.rename(columns={'price': 'price_usd'}, inplace=True)

    return df_proc


def limpieza_sencilla(df: pd.DataFrame, log_collector: list = None) -> pd.DataFrame:
    """Aplica limpieza genérica: fillna + reemplazo outliers (+-2 std) + fillna."""
    if log_collector is not None: log_collector.append("Aplicando Limpieza Sencilla...")
    df_clean = df.copy()
    df_clean = df_clean.bfill().ffill()
    numeric_cols = get_numeric_columns(df_clean)
    for col in numeric_cols:
        if col not in df_clean.columns or df_clean[col].nunique() < 2: continue
        s_num = pd.to_numeric(df_clean[col], errors='coerce').dropna()
        if s_num.empty: continue
        mean = s_num.mean(); std = s_num.std()
        if pd.isna(std) or std == 0: continue
        lower_limit = mean - 2 * std; upper_limit = mean + 2 * std
        mask_outliers = (df_clean[col] < lower_limit) | (df_clean[col] > upper_limit)
        if mask_outliers.sum() > 0:
            if log_collector is not None: log_collector.append(f"  - Columna '{col}': {mask_outliers.sum()} outliers reemplazados.")
            df_clean.loc[mask_outliers, col] = np.nan
    df_clean = df_clean.bfill().ffill()
    return df_clean


def limpieza_robusta(df: pd.DataFrame, log_collector: list = None) -> pd.DataFrame:
    """Aplica la lógica de limpieza específica por variable (imputación + outliers +-3 std)."""
    if log_collector is not None: log_collector.append("Aplicando Limpieza Robusta ...")
    df_clean = df.copy()

    text_imputations = {
        "description": "NO DESCRIPTION", "neighborhood_overview": "NO NEIGHBORHOOD_OVERVIEW",
        "picture_url": "NO PICTURE URL", "host_name": "NO HOST NAME", "host_location": "UNKNOWN",
        "host_about": "HOST ABOUT NOT PROVIDED", "host_thumbnail_url": "NO PROVIDED URL",
        "host_picture_url": "NO PROVIDED URL", "host_neighbourhood": "UNKNOWN",
        "host_is_superhost": "SIN_INFO", "license": "SL222222X2XX2XXXX2"
    }
    for col, sentinel in text_imputations.items():
        if col in df_clean.columns: df_clean[col] = df_clean[col].fillna(sentinel)
    cols_moda_host = [
        "host_since", "host_listings_count", "host_total_listings_count", "host_verifications",
        "host_has_profile_pic", "host_identity_verified"
    ]
    for col in cols_moda_host:
        if col in df_clean.columns and df_clean[col].isna().any():
            if not df_clean[col].dropna().empty: moda = df_clean[col].mode()[0]; df_clean[col] = df_clean[col].fillna(moda)
    for col in ['host_response_time', 'host_response_rate', 'host_acceptance_rate']:
        if col not in df_clean.columns: continue
        if col == 'host_response_time': df_clean[col] = df_clean[col].fillna("N/A")
        else:
            numeric_col = pd.to_numeric(df_clean[col].astype(str).str.replace("%", "", regex=False), errors='coerce')
            numeric_col = numeric_col / 100.0; df_clean[col] = numeric_col.fillna(222)
    if "neighbourhood" in df_clean.columns and "neighbourhood_cleansed" in df_clean.columns:
         if not df_clean[["neighbourhood", "neighbourhood_cleansed"]].dropna().empty:
            mapping = df_clean.dropna(subset=["neighbourhood", "neighbourhood_cleansed"]).drop_duplicates(subset=["neighbourhood_cleansed"]).set_index("neighbourhood_cleansed")["neighbourhood"].to_dict()
            def fill_neighbourhood(row):
                if pd.isna(row["neighbourhood"]): return mapping.get(row["neighbourhood_cleansed"], row["neighbourhood"])
                return row["neighbourhood"]
            df_clean["neighbourhood"] = df_clean.apply(fill_neighbourhood, axis=1)
    if "neighbourhood" in df_clean.columns and df_clean["neighbourhood"].isna().any():
         if not df_clean["neighbourhood"].dropna().empty: moda_neigh = df_clean["neighbourhood"].mode()[0]; df_clean["neighbourhood"] = df_clean["neighbourhood"].fillna(moda_neigh)
    if "bathrooms" in df_clean.columns and "bathrooms_text" in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean["bathrooms"]): df_clean["bathrooms"] = pd.to_numeric(df_clean["bathrooms"], errors='coerce')
        if pd.api.types.is_string_dtype(df_clean["bathrooms_text"]):
            half_bath_variants = ["shared half-bath", "private half-bath", "half-bath"]; mask_half_bath = df_clean["bathrooms_text"].fillna('').str.strip().str.lower().isin(half_bath_variants)
            df_clean.loc[mask_half_bath, "bathrooms"] = 0.5
        if df_clean["bathrooms"].isna().any() and not df_clean["bathrooms"].dropna().empty: moda_bathrooms = df_clean["bathrooms"].mode(dropna=True).iloc[0]; df_clean["bathrooms"] = df_clean["bathrooms"].fillna(moda_bathrooms)
    if "bathrooms_text" in df_clean.columns: df_clean["bathrooms_text"] = df_clean["bathrooms_text"].fillna("1 bath")
    for col, method in [("bedrooms", "median"), ("beds", "mode")]:
         if col in df_clean.columns and df_clean[col].isna().any():
              s_num = pd.to_numeric(df_clean[col], errors='coerce')
              if not s_num.dropna().empty:
                   if method == "median": value_to_fill = s_num.median(skipna=True)
                   else: value_to_fill = s_num.mode(dropna=True).iloc[0]
                   df_clean[col] = df_clean[col].fillna(value_to_fill)

    price_col_name = 'price_usd' if 'price_usd' in df_clean.columns else 'price'
    if price_col_name in df_clean.columns and df_clean[price_col_name].isnull().any():
        if not df_clean[price_col_name].dropna().empty:
            media_price = df_clean[price_col_name].mean(skipna=True); df_clean[price_col_name] = df_clean[price_col_name].fillna(media_price)

    if "calendar_updated" in df_clean.columns: df_clean = df_clean.drop(columns=["calendar_updated"])
    if "has_availability" in df_clean.columns and df_clean["has_availability"].isna().any():
        if not df_clean["has_availability"].dropna().empty: moda_has_avail = df_clean["has_availability"].mode(dropna=True).iloc[0]; df_clean["has_availability"] = df_clean["has_availability"].fillna(moda_has_avail)
    if "estimated_revenue_l365d" in df_clean.columns: df_clean["estimated_revenue_l365d"] = df_clean["estimated_revenue_l365d"].fillna(0)
    cols_fechas_review = ["first_review", "last_review"]
    cols_num_review = ["reviews_per_month", "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value"]
    for col in cols_fechas_review:
        if col in df_clean.columns: df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce").fillna(pd.Timestamp("2222-01-01"))
    for col in cols_num_review:
        if col in df_clean.columns: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(-222)

    sentinel_values = {"host_response_rate": 222, "host_acceptance_rate": 222, "reviews_per_month": -222, "review_scores_rating": -222, "review_scores_accuracy": -222, "review_scores_cleanliness": -222, "review_scores_checkin": -222, "review_scores_communication": -222, "review_scores_location": -222, "review_scores_value": -222, "estimated_revenue_l365d": 0}
    exclude_cols = ["id", "host_id", "latitude", "longitude", "scrape_id"]
    num_cols_outliers = [c for c in get_numeric_columns(df_clean) if c not in exclude_cols]
    for col in num_cols_outliers:
        if col not in df_clean.columns: continue
        s_num = pd.to_numeric(df_clean[col], errors='coerce'); sentinel = sentinel_values.get(col)
        mask_valid = s_num.notna()
        if sentinel is not None and col != "estimated_revenue_l365d": mask_valid = mask_valid & (s_num != sentinel)
        vals = s_num[mask_valid]
        if vals.size < 2: continue
        mean, std = vals.mean(), vals.std()
        if std == 0 or np.isnan(std): continue
        lower, upper = mean - 3 * std, mean + 3 * std
        mask_outliers_in_valid = ((vals < lower) | (vals > upper)); outlier_indices = vals[mask_outliers_in_valid].index
        if not outlier_indices.empty:
            num_outliers = len(outlier_indices)
            if log_collector is not None: log_collector.append(f"  - Columna '{col}': {num_outliers} outliers reemplazados (Limp. Robusta).")
            df_clean.loc[outlier_indices, col] = np.nan
            valid_non_outlier_vals = vals[~mask_outliers_in_valid]; valor = np.nan
            if not valid_non_outlier_vals.empty:
                if col == price_col_name: valor = valid_non_outlier_vals.mean()
                elif col == "bedrooms": valor = valid_non_outlier_vals.median()
                elif col in ["bathrooms", "beds", "host_listings_count", "host_total_listings_count"]:
                     if not valid_non_outlier_vals.mode().empty: valor = valid_non_outlier_vals.mode().iloc[0]
                elif col == "estimated_revenue_l365d": valor = 0
                elif col in sentinel_values and sentinel is not None: valor = sentinel
                else: valor = valid_non_outlier_vals.mean()
            if pd.notna(valor): df_clean[col] = df_clean[col].fillna(valor)

    for col in df_clean.select_dtypes(include=np.number).columns: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.bfill().ffill()
    return df_clean

#####################################################
# Funciones Helper para Modelado
#####################################################
def calculate_adj_r2(r2, n, p):
    """Calcula el R² Ajustado."""
    if pd.isna(r2) or n <= p + 1: return np.nan
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def style_airbnb_plot(ax, fig):
    """Aplica un estilo visual a los gráficos de Matplotlib."""
    BG_COLOR="#FFFFFF"; PRIMARY_COLOR="#FF5A5F"; SECONDARY_COLOR="#00A699"; TEXT_COLOR="#484848"; GRID_COLOR="#EBEBEB"
    try:
        fig.set_facecolor(BG_COLOR); ax.set_facecolor(BG_COLOR); ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR); ax.yaxis.label.set_color(TEXT_COLOR); ax.tick_params(colors=TEXT_COLOR, which='both')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(TEXT_COLOR); ax.spines['bottom'].set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5, zorder=-1)
    except AttributeError: pass
    return PRIMARY_COLOR, SECONDARY_COLOR, TEXT_COLOR

def get_top10_categoricas(df: pd.DataFrame):
    """Obtiene las 10 mejores columnas categóricas (baja cardinalidad)."""
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist(); meta = []
    for c in cat_cols:
        if c in df.columns:
            nuniq = df[c].nunique(dropna=True)
            if 1 < nuniq <= 10: meta.append((c, nuniq, df[c].isna().sum(), df[c].notna().sum()))
    if not meta: return []
    meta_df = pd.DataFrame(meta, columns=["col","n_levels","n_null","n_nonnull"])
    meta_df = meta_df.sort_values(["n_nonnull","n_levels"], ascending=[False, True]).head(10)
    return meta_df["col"].tolist()

def top10_corr_for(num_df: pd.DataFrame, y: str) -> pd.DataFrame:
    """Calcula el Top 10 de correlación absoluta para una variable 'y'."""
    if y not in num_df.columns: return pd.DataFrame()
    try:
        valid_cols = num_df.select_dtypes(include=np.number).columns
        corr_series = num_df[valid_cols].corr(numeric_only=True)[y].drop(y, errors='ignore').dropna()
        if corr_series.empty: return pd.DataFrame()
        corr = corr_series.abs().sort_values(ascending=False).head(10)
        return pd.DataFrame({"Variable": corr.index, "Correlación_abs": corr.values})
    except Exception: return pd.DataFrame()

def top_k_no_other_bins(frame, target, k=5, base_exclude=set(), derived_bins=set(), extra_exclusions=set(), min_var=1e-12):
    """Encuentra las K mejores correlaciones para regresión logística."""
    all_excl = set(base_exclude) | extra_exclusions | (derived_bins - {target}) | {target}
    if target not in frame.columns: return pd.DataFrame()
    numeric_cols = frame.select_dtypes(include=np.number).columns
    cols = [c for c in numeric_cols if c not in all_excl and c in frame.columns]
    rows = []
    for c in cols:
        pair = frame[[target, c]].dropna();
        if pair.shape[0] < 3: continue
        try:
             target_var = pair[target].var(); c_var = pair[c].var()
             if pd.isna(target_var) or pd.isna(c_var) or target_var < min_var or c_var < min_var: continue
        except Exception: continue
        try:
             r = pair[target].corr(pair[c])
             if pd.notna(r): rows.append((c, r, abs(r), pair.shape[0]))
        except Exception: continue
    out = pd.DataFrame(rows, columns=['Variable','Correlación','Abs_Corr','N'])
    if out.empty: return out
    return (out.sort_values(['Abs_Corr', 'N'], ascending=[False, False])
               .head(k).drop(columns=['Abs_Corr']).reset_index(drop=True))

#####################################################
# Definiciones de Páginas de la App
#####################################################

def pagina_limpieza_analisis(df_original, df, metodo_limpieza, log_mensajes: list):
    """Renderiza la página de limpieza y análisis exploratorio."""
    if log_mensajes:
        st.subheader("Registro de Limpieza"); st.code("\n".join(log_mensajes), language="text"); st.markdown("<hr>", unsafe_allow_html=True)
    if not df_original.empty:
        if metodo_limpieza != "Sin Limpieza":
            st.header("Verificación de Limpieza de Datos"); col1, col2 = st.columns(2)
            with col1: st.subheader("Antes de la Limpieza"); st.info(f"Dimensiones: {df_original.shape}"); st.dataframe(df_original.head())
            with col2: st.subheader("Después de la Limpieza"); st.info(f"Dimensiones: {df.shape}"); st.dataframe(df.head())
        else: st.header("Datos Cargados"); st.info(f"Mostrando datos combinados (Dimensiones: {df.shape})"); st.dataframe(df.head())
        st.header("Análisis Exploratorio (Ejemplo)");
        price_col_histo = 'price_usd' if 'price_usd' in df.columns else 'price' if 'price' in df.columns else None
        columnas_numericas = get_numeric_columns(df)
        if columnas_numericas:
            color_col = "origen_archivo" if "origen_archivo" in df.columns and df["origen_archivo"].nunique() > 1 else None
            default_histo_col = price_col_histo if price_col_histo in columnas_numericas else columnas_numericas[0]
            col_para_histo = st.selectbox("Selecciona columna numérica para histograma:", columnas_numericas, index=(columnas_numericas.index(default_histo_col) if default_histo_col in columnas_numericas else 0), key="limpieza_histo_select")
            if col_para_histo:
                try:
                    df_plot = df.dropna(subset=[col_para_histo])
                    if not df_plot.empty:
                        fig = px.histogram(df_plot, x=col_para_histo, color=color_col, barmode="overlay", title=f"Histograma de '{col_para_histo}' (Post-Limpieza)")
                        fig.update_layout(xaxis_title=col_para_histo, yaxis_title="Conteo"); st.plotly_chart(fig, use_container_width=True)
                    else: st.info(f"No hay datos válidos para histograma de '{col_para_histo}'.")
                except Exception as e: st.error(f"No se pudo graficar '{col_para_histo}'. Error: {e}")
        else: st.warning("No hay columnas numéricas válidas para analizar.")
        st.header("Visualización Personalizada de Datos")
        if not df.empty:
            todas_las_columnas = df.columns.tolist()
            if 'origen_archivo' in todas_las_columnas: todas_las_columnas.insert(0, todas_las_columnas.pop(todas_las_columnas.index('origen_archivo')))
            default_cols = todas_las_columnas[:min(5, len(todas_las_columnas))]
            columnas_para_ver = st.multiselect("Selecciona columnas para visualizar:", options=todas_las_columnas, default=default_cols, key="limpieza_table_multi")
            if columnas_para_ver: st.dataframe(df[columnas_para_ver])
            else: st.info("Selecciona >= 1 columna para crear tabla.")

def pagina_extraccion_caracteristicas(df):
    """Renderiza el dashboard de extracción de características estilo Airbnb."""
    st.markdown("""<style>
        [data-testid="stMetric"] { background-color: var(--secondary-background-color); border: 1px solid var(--streamlit-gray-300); border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        [data-testid="stMetricValue"] { font-size: 1.75rem; } [data-testid="stMetricLabel"] { font-size: 1rem; }
        </style>""", unsafe_allow_html=True)
    if df.empty: st.warning("No hay datos limpios para analizar."); return

    st.markdown("### Filtros del Dashboard")
    all_origins_initial = df['origen_archivo'].unique() if 'origen_archivo' in df.columns else []
    is_multi_db_initial = len(all_origins_initial) > 1
    col_f1, col_f2 = st.columns(2); df_filtrado = df.copy(); selected_origins = all_origins_initial
    if is_multi_db_initial:
        with col_f1:
            selected_origins = st.multiselect("Filtrar por Base de Datos:", options=all_origins_initial, default=all_origins_initial, key="extract_origin_multi")
            if selected_origins: df_filtrado = df_filtrado[df_filtrado['origen_archivo'].isin(selected_origins)]
            else: st.warning("Selecciona al menos una base de datos."); df_filtrado = pd.DataFrame(columns=df.columns)
    with col_f2:
        if 'room_type' in df_filtrado.columns:
            room_types_disponibles = sorted(df_filtrado['room_type'].dropna().unique().tolist())
            tipos_seleccionados = st.multiselect("Filtrar por tipo de habitación:", options=room_types_disponibles, default=room_types_disponibles, key="extract_room_multi")
            if tipos_seleccionados: df_filtrado = df_filtrado[df_filtrado['room_type'].isin(tipos_seleccionados)]
            elif room_types_disponibles: st.warning("Selecciona al menos un tipo de habitación."); df_filtrado = pd.DataFrame(columns=df.columns)
        elif not df_filtrado.empty: st.info("No se encontró 'room_type' para filtrar.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Métricas Generales (Según Filtros)")
    col1, col2, col3, col4, col5 = st.columns(5)
    num_propiedades = len(df_filtrado); col1.metric("🏠 Propiedades", f"{num_propiedades:,}")
    price_col_kpi = 'price_usd' if 'price_usd' in df_filtrado.columns else 'price'
    precio_promedio = pd.to_numeric(df_filtrado.get(price_col_kpi), errors='coerce').mean() if not df_filtrado.empty else 0; col2.metric("💵 Precio Prom. (USD)", f"${precio_promedio:,.2f}" if pd.notna(precio_promedio) else "N/A")
    num_reviews = pd.to_numeric(df_filtrado.get('number_of_reviews'), errors='coerce').sum() if not df_filtrado.empty else 0; col3.metric("✍️ Reseñas", f"{int(num_reviews):,}" if pd.notna(num_reviews) else "N/A")
    disponibilidad = pd.to_numeric(df_filtrado.get('availability_365'), errors='coerce').mean() if not df_filtrado.empty else 0; col4.metric("🗓️ Disp. (365d)", f"{disponibilidad:.1f}d" if pd.notna(disponibilidad) else "N/A")
    zona_top = "N/D";
    if 'neighbourhood_cleansed' in df_filtrado.columns and not df_filtrado.empty: mode_result = df_filtrado['neighbourhood_cleansed'].mode();
    if not mode_result.empty: zona_top = mode_result[0]
    col5.metric("📍 Zona Top", zona_top)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Distribución Geográfica de Propiedades")
    if not is_multi_db_initial:
        lat_col, lon_col = 'latitude', 'longitude'
        if lat_col in df_filtrado.columns and lon_col in df_filtrado.columns:
            map_data = df_filtrado[[lat_col, lon_col, 'origen_archivo']].copy(); map_data[lat_col] = pd.to_numeric(map_data[lat_col], errors='coerce'); map_data[lon_col] = pd.to_numeric(map_data[lon_col], errors='coerce'); map_data.dropna(subset=[lat_col, lon_col], inplace=True); map_data.rename(columns={lat_col: 'lat', lon_col: 'lon'}, inplace=True)
            if not map_data.empty: st.map(map_data, zoom=10)
            else: st.info("No hay datos de ubicación válidos (después de filtrar).")
        else: st.info(f"Se necesitan '{lat_col}' y '{lon_col}'.")
    else: st.info("Mapa oculto cuando se cargan múltiples bases.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Análisis de Características")
    color_airbnb = '#FF5A5F' # Color base para gráfico único

    variables_a_graficar = {
        "Tipo de Propiedad": "property_type", "Tipo de Habitación": "room_type",
        "¿Es Superhost?": "host_is_superhost", "Tiempo de Respuesta del Host": "host_response_time",
        "Identidad Verificada": "host_identity_verified", "Foto de Perfil": "host_has_profile_pic",
        "Número de Baños": "bathrooms", "Número de Camas": "beds"
    }
    available_vars = {k:v for k,v in variables_a_graficar.items() if v in df_filtrado.columns}
    if available_vars:
        seleccion_usuario = st.selectbox("Selecciona característica:", options=list(available_vars.keys()), key="extract_feature_select")
        columna_seleccionada = available_vars[seleccion_usuario]
        if not df_filtrado.empty and columna_seleccionada in df_filtrado.columns:
            show_separate_charts = len(selected_origins) > 1

            # Definir paletas de colores Airbnb
            color_palettes = [
                px.colors.sequential.Reds_r,
                px.colors.sequential.Tealgrn,
                px.colors.sequential.OrRd,
                px.colors.sequential.Blues_r,
            ]

            if show_separate_charts:
                chart_cols = st.columns(len(selected_origins))
                for i, origin in enumerate(selected_origins):
                    with chart_cols[i]:
                        st.markdown(f"##### {origin.replace('.csv', '')}")
                        df_origin = df_filtrado[df_filtrado['origen_archivo'] == origin].copy(); df_plot_origin = df_origin[[columna_seleccionada]].dropna()
                        if not df_plot_origin.empty:
                            fig_origin = None
                            try:
                                current_palette = color_palettes[i % len(color_palettes)]
                                bar_color = px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)]

                                if columna_seleccionada in ["host_is_superhost", "host_identity_verified", "host_has_profile_pic", "room_type"]:
                                    counts = df_plot_origin[columna_seleccionada].value_counts()
                                    if len(counts) > 5: top_4 = counts.nlargest(4); otros = pd.Series(counts.nsmallest(len(counts) - 4).sum(), index=['Otros']); counts = pd.concat([top_4, otros])
                                    fig_origin = px.pie(values=counts.values, names=counts.index, title=f"'{seleccion_usuario}'", color_discrete_sequence=current_palette)
                                    fig_origin.update_traces(textposition='inside', textinfo='percent+label', showlegend=False,textfont_size=20); fig_origin.update_layout(title_font_size=16, title_x=0.5, margin=dict(t=50, b=20, l=20, r=20))
                                elif columna_seleccionada in ["property_type", "host_response_time"]:
                                    counts = df_plot_origin[columna_seleccionada].value_counts().nlargest(10).sort_values()
                                    fig_origin = px.bar(x=counts.values, y=counts.index, orientation='h', title=f"Top 10 '{seleccion_usuario}'", labels={'x': 'Cant.', 'y': ''}, color_discrete_sequence=[bar_color]) # Color aplicado
                                    fig_origin.update_layout(title_font_size=14, title_x=0.5, yaxis={'categoryorder':'total ascending'})
                                elif columna_seleccionada in ["bathrooms", "beds"]:
                                    df_plot_origin[columna_seleccionada] = df_plot_origin[columna_seleccionada].astype(str); counts = df_plot_origin[columna_seleccionada].value_counts().nlargest(10).sort_index()
                                    fig_origin = px.bar(x=counts.index, y=counts.values, title=f"'{seleccion_usuario}'", labels={'x': seleccion_usuario, 'y': 'Cant.'}, color_discrete_sequence=[bar_color]) # Color aplicado
                                    try: fig_origin.update_xaxes(categoryorder='array', categoryarray=sorted(counts.index, key=float))
                                    except: fig_origin.update_xaxes(categoryorder='total descending')
                                    fig_origin.update_layout(title_font_size=14, title_x=0.5)
                                if fig_origin: st.plotly_chart(fig_origin, use_container_width=True)
                                else: st.info(f"Tipo gráfico no definido.")
                            except Exception as e: st.error(f"Error al graficar: {e}")
                        else: st.info(f"No hay datos válidos.")
            else: # Gráfico único
                df_plot = df_filtrado[[columna_seleccionada, 'origen_archivo']].copy().dropna(subset=[columna_seleccionada])
                if not df_plot.empty:
                    fig = None
                    try:
                        color_dinamico_single = "origen_archivo" if 'origen_archivo' in df_plot.columns and df_plot['origen_archivo'].nunique() > 1 else None
                        single_chart_palette = px.colors.sequential.Reds_r

                        if columna_seleccionada in ["host_is_superhost", "host_identity_verified", "host_has_profile_pic", "room_type"]:
                            if color_dinamico_single: counts = df_plot.groupby([columna_seleccionada, color_dinamico_single]).size().reset_index(name='count'); fig = px.bar(counts, x=columna_seleccionada, y='count', color=color_dinamico_single, title=f"Distribución de '{seleccion_usuario}' por Base", barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
                            else: counts = df_plot[columna_seleccionada].value_counts(); fig = px.pie(values=counts.values, names=counts.index, title=f"Distribución de '{seleccion_usuario}'", color_discrete_sequence=single_chart_palette); fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20)
                        elif columna_seleccionada in ["property_type", "host_response_time"]:
                            if color_dinamico_single: top_categories = df_plot[columna_seleccionada].value_counts().nlargest(10).index; df_top = df_plot[df_plot[columna_seleccionada].isin(top_categories)]; counts = df_top.groupby([columna_seleccionada, color_dinamico_single]).size().reset_index(name='count'); fig = px.bar(counts, y=columna_seleccionada, x='count', color=color_dinamico_single, orientation='h', title=f"Top 10 de '{seleccion_usuario}' por Base", barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel); fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            else: counts = df_plot[columna_seleccionada].value_counts().nlargest(10).sort_values(); fig = px.bar(x=counts.values, y=counts.index, orientation='h', title=f"Top 10 de '{seleccion_usuario}'", labels={'x': 'Cantidad', 'y': seleccion_usuario}, color_discrete_sequence=[color_airbnb])
                        elif columna_seleccionada in ["bathrooms", "beds"]:
                            df_plot[columna_seleccionada] = df_plot[columna_seleccionada].astype(str)
                            if color_dinamico_single: top_categories = df_plot[columna_seleccionada].value_counts().nlargest(10).index; df_top = df_plot[df_plot[columna_seleccionada].isin(top_categories)]; counts = df_top.groupby([columna_seleccionada, color_dinamico_single]).size().reset_index(name='count'); fig = px.bar(counts, x=columna_seleccionada, y='count', color=color_dinamico_single, title=f"Distribución de '{seleccion_usuario}' por Base", barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
                            else: counts = df_plot[columna_seleccionada].value_counts().nlargest(10).sort_index(); fig = px.bar(x=counts.index, y=counts.values, title=f"Distribución de '{seleccion_usuario}'", labels={'x': seleccion_usuario, 'y': 'Propiedades'}, color_discrete_sequence=[color_airbnb])
                            try: fig.update_xaxes(categoryorder='array', categoryarray=sorted(df_plot[columna_seleccionada].unique(), key=float))
                            except: fig.update_xaxes(categoryorder='total descending')
                        if fig: st.plotly_chart(fig, use_container_width=True)
                        else: st.info(f"Tipo gráfico no definido.")
                    except Exception as e: st.error(f"Error al generar gráfico: {e}")
                else: st.info(f"No hay datos válidos para '{seleccion_usuario}'.")
        else: st.info(f"Columna '{columna_seleccionada}' no disponible.")
    else: st.warning("No hay características válidas disponibles para graficar.")

def pagina_modelado(df):
    """Renderiza la página de Modelado (Correlaciones, Regresiones)."""
    if df.empty: st.warning("No hay datos limpios para analizar."); return
    df_clean = df.copy()
    numeric_sentinels_to_nan = {'host_response_rate': 222, 'host_acceptance_rate': 222, 'reviews_per_month': -222, 'review_scores_rating': -222, 'review_scores_accuracy': -222, 'review_scores_cleanliness': -222, 'review_scores_checkin': -222, 'review_scores_communication': -222, 'review_scores_location': -222, 'review_scores_value': -222}
    for col, sentinel_val in numeric_sentinels_to_nan.items():
        if col in df_clean.columns: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce'); df_clean[col] = df_clean[col].replace(sentinel_val, np.nan)
    if 'origen_archivo' not in df_clean.columns: st.error("Falta la columna 'origen_archivo'."); return
    all_origins = sorted(df_clean['origen_archivo'].unique()); is_multi_db = len(all_origins) > 1
    ID_COLS_TO_EXCLUDE = {'id', 'scrape_id', 'host_id'}
    for col in df_clean.select_dtypes(include=[np.number]).columns: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    num_df_all = df_clean.select_dtypes(include=[float,int]).replace([np.inf,-np.inf], np.nan)
    num_cols_keep = [c for c in num_df_all.columns if c not in ID_COLS_TO_EXCLUDE]; num_df = num_df_all[num_cols_keep].dropna(axis=1, how="all")
    top10_cats = get_top10_categoricas(df_clean)
    df_bin = df_clean.copy() # Preparación Variables Binarias
    if 'accommodates' in df_bin.columns: acc_num = pd.to_numeric(df_bin['accommodates'], errors='coerce'); df_bin['y_accommodates_le4'] = np.where(acc_num <= 4, 1, np.where(acc_num > 4, 0, np.nan))
    if 'beds' in df_bin.columns: beds_num = pd.to_numeric(df_bin['beds'], errors='coerce'); df_bin['y_beds_le3'] = np.where(beds_num <= 3, 1, np.where(beds_num > 3, 0, np.nan))
    if 'host_is_superhost' in df_bin.columns: df_bin['bin_host_is_superhost'] = df_bin['host_is_superhost'].map({'t':1, 'f':0, 'True':1, 'False':0, True:1, False:0}).fillna(np.nan)
    if 'host_has_profile_pic' in df_bin.columns: df_bin['bin_host_has_profile_pic'] = df_bin['host_has_profile_pic'].map({'t':1, 'f':0, 'True':1, 'False':0, True:1, False:0}).fillna(np.nan)
    if 'host_identity_verified' in df_bin.columns: df_bin['bin_host_identity_verified'] = df_bin['host_identity_verified'].map({'t':1, 'f':0, 'True':1, 'False':0, True:1, False:0}).fillna(np.nan)
    if 'instant_bookable' in df_bin.columns: df_bin['bin_instant_bookable'] = df_bin['instant_bookable'].map({'t':1, 'f':0, 'True':1, 'False':0, True:1, False:0}).fillna(np.nan)
    if 'neighbourhood_group_cleansed' in df_bin.columns: df_bin['bin_neigh_isole'] = df_bin['neighbourhood_group_cleansed'].apply(lambda x: 1 if x == 'Isole' else 0 if pd.notna(x) else np.nan)
    if 'room_type' in df_bin.columns: df_bin['bin_room_entire'] = np.where(df_bin['room_type']=='Entire home/apt', 1, np.where(df_bin['room_type'].isin(['Hotel room','Private room','Shared room']), 0, np.nan))
    if 'host_response_time' in df_bin.columns: fast = {'within an hour', 'within a few hours'}; slow = {'within a day', 'a few days or more'}; df_bin['bin_response_fast'] = np.where(df_bin['host_response_time'].astype(str).isin(fast), 1, np.where(df_bin['host_response_time'].astype(str).isin(slow), 0, np.nan))
    if 'host_verifications' in df_bin.columns:
        def parse_verif_exact_triple(x):
            if pd.isna(x): return np.nan;
            try:
                if not isinstance(x, str): x = str(x); x = x.replace("'", '"'); lst = ast.literal_eval(x)
                if not isinstance(lst, list): return 0; s_lower = {item.lower() for item in lst if isinstance(item, str)}; required_items = {'email', 'phone', 'work_email'}; return 1 if required_items.issubset(s_lower) else 0
            except Exception: return 0
        df_bin['bin_verif_strong'] = df_bin['host_verifications'].apply(parse_verif_exact_triple)
    target_bins_base = ['y_accommodates_le4', 'y_beds_le3', 'bin_host_is_superhost', 'bin_host_has_profile_pic', 'bin_host_identity_verified', 'bin_instant_bookable', 'bin_neigh_isole', 'bin_room_entire', 'bin_response_fast', 'bin_verif_strong']
    target_bins = [tb for tb in target_bins_base if tb in df_bin.columns]; derived_bins = set(target_bins); extra_exclusions = ID_COLS_TO_EXCLUDE
    forbidden_map = {'y_accommodates_le4': {'accommodates', 'y_beds_le3'}, 'y_beds_le3': {'beds', 'y_accommodates_le4'}, 'bin_host_is_superhost': set(), 'bin_host_has_profile_pic': set(), 'bin_host_identity_verified': set(), 'bin_instant_bookable': set(), 'bin_neigh_isole': {'neighbourhood_group_cleansed'}, 'bin_room_entire': {'room_type'}, 'bin_response_fast': {'host_response_time'}, 'bin_verif_strong': {'host_verifications'}}
    target_titles = {'y_accommodates_le4': 'Capacidad <= 4 (1=sí)', 'y_beds_le3': 'Camas <= 3 (1=sí)', 'bin_host_is_superhost': 'Superhost (1=sí)', 'bin_host_has_profile_pic': 'Foto perfil (1=sí)', 'bin_host_identity_verified': 'ID Verificada (1=sí)', 'bin_instant_bookable': 'Reserva Instant. (1=sí)', 'bin_neigh_isole': 'Zona Isole (1=Isole)', 'bin_room_entire': 'Tipo Entire home (1=sí)', 'bin_response_fast': 'Respuesta rápida (1=sí)', 'bin_verif_strong': 'Triple Verif. (1=sí)'}

    tab2, tab3, tab4 = st.tabs(["🌡️ Correlaciones","📈 Regresiones (Lineales/No lineal)","🧪 Regresión Logística"])

    with tab2: # Correlaciones
        st.subheader("Correlaciones y visualizaciones")
        if num_df.shape[1] >= 2:
            targets_num_corr = sorted([c for c in num_df.columns if num_df[c].notna().sum() > 20])
            if targets_num_corr:
                y_sel = st.selectbox("Variable objetivo (numérica)", targets_num_corr, index=0, key="corr_y_sel")
                top_10_df = top10_corr_for(num_df, y_sel)
                if not top_10_df.empty:
                    st.dataframe(top_10_df)
                    if st.toggle("Mostrar HEATMAP de correlación (Top 10 + Objetivo)", key="corr_heatmap_toggle"):
                        top_10_vars = top_10_df['Variable'].tolist(); heatmap_vars = [y_sel] + top_10_vars; heatmap_vars = [var for var in heatmap_vars if var in num_df.columns]
                        if len(heatmap_vars) > 1:
                            fig_h, ax_h = plt.subplots(figsize=(10,8)); corr_subset = num_df[heatmap_vars].corr(numeric_only=True)
                            sns.heatmap(corr_subset, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax_h); ax_h.set_title(f"Heatmap ({y_sel} y Top 10)"); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout(); st.pyplot(fig_h, use_container_width=True)
                        else: st.warning("No hay suficientes variables para heatmap.")
                else: st.warning(f"No se pudo calcular correlación para '{y_sel}'.")
                group_sel = st.selectbox("Variable categórica (para BOX PLOT)", ["(ninguna)"] + top10_cats, index=0, key="corr_boxplot_group")
                if group_sel != "(ninguna)":
                    if y_sel in df_clean.columns:
                        sub_bp = df_clean[[group_sel, y_sel, 'origen_archivo']].dropna(subset=[group_sel, y_sel])
                        if not sub_bp.empty:
                            fig_bp, ax_bp = plt.subplots(figsize=(8,5)); AIRBNB_RED, _, _ = style_airbnb_plot(ax_bp, fig_bp); color_bp = "origen_archivo" if is_multi_db else None
                            try: sns.boxplot(data=sub_bp, x=group_sel, y=y_sel, hue=color_bp, ax=ax_bp, palette="viridis" if color_bp else [AIRBNB_RED]); ax_bp.set_title(f"{y_sel} por {group_sel}"); ax_bp.tick_params(axis='x', rotation=15); style_airbnb_plot(ax_bp, fig_bp); st.pyplot(fig_bp, use_container_width=True)
                            except Exception as e: st.error(f"Error en boxplot: {e}")
                        else: st.info(f"No hay datos para boxplot.")
                    else: st.warning(f"'{y_sel}' no encontrada para boxplot.")
        else: st.warning("No hay suficientes columnas numéricas para correlaciones.")

    with tab3: # Regresiones
        st.subheader("Modelado: lineal simple, lineal múltiple y no lineal")
        if num_df.shape[1] >= 2:
            targets_num_reg = sorted([c for c in num_df.columns if num_df[c].notna().sum() > 20])
            # --- Lineal simple ---
            st.markdown("### 🔹 Regresión lineal **simple**"); y1 = st.selectbox("Objetivo (lineal simple)", targets_num_reg, key="lin_simple_y")
            available_preds_s = [c for c in num_df.columns if c != y1 and num_df[c].notna().sum() > 20]
            if available_preds_s:
                valid_corr_targets = [p for p in available_preds_s if p in num_df.columns]; corr_abs = pd.Series(dtype=float)
                if y1 in num_df.columns and valid_corr_targets: corr_abs = num_df.corr(numeric_only=True)[y1].drop(y1, errors='ignore').loc[valid_corr_targets].abs().sort_values(ascending=False)
                pred_default = corr_abs.index[0] if not corr_abs.empty else available_preds_s[0]; x1 = st.selectbox("Predictor (X)", available_preds_s, index=(available_preds_s.index(pred_default) if pred_default in available_preds_s else 0), key="lin_simple_x")
                if x1:
                    sub_s = df_clean[[x1, y1, 'origen_archivo']].copy()
                    if len(sub_s[[x1, y1]].dropna()) >= 10:
                        X_global = sm.add_constant(sub_s[[x1]], has_constant='add'); y_global_true = sub_s[y1]
                        try:
                            modelo = sm.OLS(y_global_true, X_global, missing='drop').fit(); st.markdown("### Resultados del Modelo Global"); st.markdown(f"```text\n{modelo.summary()}\n```")
                            fitted_data_x = X_global.loc[modelo.fittedvalues.index, x1]; min_x, max_x = fitted_data_x.min(), fitted_data_x.max()
                            if pd.notna(min_x) and pd.notna(max_x):
                                xx = np.linspace(min_x, max_x, 200); Xp = pd.DataFrame({"const":1.0, x1:xx}); yhat_curve = modelo.predict(Xp); fig1, ax1 = plt.subplots(); AIRBNB_RED, AIRBNB_TEAL, _ = style_airbnb_plot(ax1, fig1); sub_plot = sub_s.dropna(subset=[x1, y1])
                                if is_multi_db:
                                    for origin in all_origins: sub_origin_plot = sub_plot[sub_plot['origen_archivo'] == origin]; ax1.scatter(sub_origin_plot[x1], sub_origin_plot[y1], alpha=0.3, label=f"Obs. ({origin})")
                                else: ax1.scatter(sub_plot[x1], sub_plot[y1], alpha=0.4, label="Observaciones", color=AIRBNB_TEAL)
                                ax1.plot(xx, yhat_curve, linewidth=2.5, label="Recta ajustada (Global)", color=AIRBNB_RED, zorder=10); ax1.set_title(f"{y1} ~ {x1} (lineal simple)"); ax1.set_xlabel(x1); ax1.set_ylabel(y1); ax1.legend(); style_airbnb_plot(ax1, fig1); st.pyplot(fig1, use_container_width=True)
                            else: st.warning(f"No se pudo graficar línea para '{x1}'.")
                            st.markdown("#### Comparación de Métricas"); comparison_data = []; p = 1; y_hat_global = modelo.fittedvalues; y_eval_global_true = y_global_true.loc[y_hat_global.index]
                            if not y_eval_global_true.empty: r2_global = r2_score(y_eval_global_true, y_hat_global); rmse_global = np.sqrt(mean_squared_error(y_eval_global_true, y_hat_global)); n_global = len(y_eval_global_true); adj_r2_global = calculate_adj_r2(r2_global, n_global, p); comparison_data.append({"Base de Datos": "Global (Combinada)", "R²": r2_global, "R² Ajustado": adj_r2_global, "RMSE": rmse_global, "n": n_global})
                            if is_multi_db:
                                for origin in all_origins:
                                    sub_origin = sub_s[sub_s['origen_archivo'] == origin]; sub_origin_cleaned = sub_origin.dropna(subset=[x1, y1]); n_origin = len(sub_origin_cleaned)
                                    if n_origin < p + 2: continue
                                    X_origin = sm.add_constant(sub_origin_cleaned[[x1]], has_constant='add'); y_true_origin = sub_origin_cleaned[y1]; y_pred_origin = modelo.predict(X_origin); r2_origin = r2_score(y_true_origin, y_pred_origin); rmse_origin = np.sqrt(mean_squared_error(y_true_origin, y_pred_origin)); adj_r2_origin = calculate_adj_r2(r2_origin, n_origin, p); comparison_data.append({"Base de Datos": origin, "R²": r2_origin, "R² Ajustado": adj_r2_origin, "RMSE": rmse_origin, "n": n_origin})
                            if comparison_data: st.dataframe(pd.DataFrame(comparison_data).set_index('Base de Datos').style.format("{:.3f}", subset=["R²", "R² Ajustado", "RMSE"]))
                            else: st.warning("No se pudieron calcular métricas.")
                        except Exception as e: st.error(f"Error al ajustar modelo lineal simple: {e}")
                    else: st.warning(f"No hay suficientes datos válidos para modelo lineal simple.")
                else: st.warning("Selecciona un predictor (X) válido.")
            else: st.warning("No hay predictores numéricos disponibles.")

            st.divider()
            # --- Lineal múltiple ---
            st.markdown("### 🔹 Regresión lineal **múltiple** (Top-5 y depuración por p<0.05)"); y2 = st.selectbox("Objetivo (lineal múltiple)", targets_num_reg, key="lin_mult_y")
            available_preds_m = [c for c in num_df.columns if c != y2 and num_df[c].notna().sum() > 20]
            if available_preds_m:
                valid_corr_targets_m = [p for p in available_preds_m if p in num_df.columns]; corr_abs2 = pd.Series(dtype=float)
                if y2 in num_df.columns and valid_corr_targets_m: corr_abs2 = num_df.corr(numeric_only=True)[y2].drop(y2, errors='ignore').loc[valid_corr_targets_m].abs().sort_values(ascending=False)
                preselect = [p for p in corr_abs2.head(5).index.tolist() if p in available_preds_m];
                if not preselect: preselect = available_preds_m[:min(5, len(available_preds_m))]
                Xs = st.multiselect("Predictores candidatos", available_preds_m, default=preselect, key="lin_mult_xs")
                x_for_plot = st.selectbox("Variable eje X para superponer predicción", Xs if Xs else [None], key="lin_mult_xplot")
                if Xs:
                    sub_m = df_clean[[y2, 'origen_archivo'] + Xs].copy()
                    if len(sub_m[[y2] + Xs].dropna()) >= 20:
                        try:
                            X_global = sm.add_constant(sub_m[Xs], has_constant='add'); y_global_true = sub_m[y2]
                            modelo_multi = sm.OLS(y_global_true, X_global, missing="drop").fit(); curr = list(Xs); final_predictors = list(Xs)
                            while True:
                                pvals = modelo_multi.pvalues.drop("const", errors="ignore")
                                if pvals.empty or pvals.max() <= 0.05: final_predictors = curr[:]; break
                                worst = pvals.idxmax()
                                if worst in curr: curr.remove(worst)
                                if not curr: final_predictors = []; break
                                X_global_curr = sm.add_constant(sub_m[curr], has_constant='add')
                                modelo_multi = sm.OLS(sub_m[y2], X_global_curr, missing="drop").fit()
                            if not final_predictors: st.warning("Ningún predictor fue significativo.")
                            else:
                                X_global_final_fit = sm.add_constant(sub_m[final_predictors], has_constant='add')
                                modelo_multi = sm.OLS(sub_m[y2], X_global_final_fit, missing="drop").fit()
                                st.markdown("### Resultados del Modelo Global (depurado p<0.05)"); st.markdown(f"```text\n{modelo_multi.summary()}\n```")
                                if x_for_plot and x_for_plot in final_predictors:
                                    fitted_indices = modelo_multi.fittedvalues.index; sub_plot_m = sub_m.loc[fitted_indices]
                                    min_x_m, max_x_m = sub_plot_m[x_for_plot].min(), sub_plot_m[x_for_plot].max()
                                    if pd.notna(min_x_m) and pd.notna(max_x_m):
                                        xx = np.linspace(min_x_m, max_x_m, 200); base = {c: sub_plot_m[c].mean() for c in final_predictors}; grid = pd.DataFrame(base, index=range(200)); grid[x_for_plot] = xx; grid = sm.add_constant(grid, has_constant='add'); yhat_curve = modelo_multi.predict(grid)
                                        fig2, ax2 = plt.subplots(); AIRBNB_RED, AIRBNB_TEAL, _ = style_airbnb_plot(ax2, fig2)
                                        if is_multi_db:
                                            for origin in all_origins: sub_origin_plot = sub_plot_m[sub_plot_m['origen_archivo'] == origin]; ax2.scatter(sub_origin_plot[x_for_plot], sub_origin_plot[y2], alpha=0.3, label=f"Obs. ({origin})")
                                        else: ax2.scatter(sub_plot_m[x_for_plot], sub_plot_m[y2], alpha=0.4, label="Observaciones", color=AIRBNB_TEAL)
                                        ax2.plot(xx, yhat_curve, linewidth=2.5, label="Predicción sobre " + x_for_plot, color=AIRBNB_RED, zorder=10); ax2.set_title(f"{y2} ~ múltiples (depurado p<0.05)"); ax2.set_xlabel(x_for_plot); ax2.set_ylabel(y2); ax2.legend(); style_airbnb_plot(ax2, fig2); st.pyplot(fig2, use_container_width=True)
                                    else: st.warning(f"No se pudo graficar la línea para '{x_for_plot}'.")
                                st.markdown("#### Comparación de Métricas"); comparison_data = []; p = len(final_predictors)
                                y_hat_global = modelo_multi.fittedvalues; y_eval_global_m_true = sub_m[y2].loc[y_hat_global.index]
                                if not y_eval_global_m_true.empty:
                                    r2_global = r2_score(y_eval_global_m_true, y_hat_global); rmse_global = np.sqrt(mean_squared_error(y_eval_global_m_true, y_hat_global)); n_global = len(y_eval_global_m_true); adj_r2_global = calculate_adj_r2(r2_global, n_global, p)
                                    comparison_data.append({"Base de Datos": "Global (Combinada)", "R²": r2_global, "R² Ajustado": adj_r2_global, "RMSE": rmse_global, "n": n_global})
                                if is_multi_db:
                                    for origin in all_origins:
                                        sub_origin = sub_m[sub_m['origen_archivo'] == origin]; sub_origin_cleaned = sub_origin.dropna(subset=[y2] + final_predictors); n_origin = len(sub_origin_cleaned)
                                        if n_origin < p + 2: continue
                                        X_origin = sm.add_constant(sub_origin_cleaned[final_predictors], has_constant='add'); y_true_origin = sub_origin_cleaned[y2]; y_pred_origin = modelo_multi.predict(X_origin)
                                        r2_origin = r2_score(y_true_origin, y_pred_origin); rmse_origin = np.sqrt(mean_squared_error(y_true_origin, y_pred_origin)); adj_r2_origin = calculate_adj_r2(r2_origin, n_origin, p)
                                        comparison_data.append({"Base de Datos": origin, "R²": r2_origin, "R² Ajustado": adj_r2_origin, "RMSE": rmse_origin, "n": n_origin})
                                if comparison_data: st.dataframe(pd.DataFrame(comparison_data).set_index('Base de Datos').style.format("{:.3f}", subset=["R²", "R² Ajustado", "RMSE"]))
                                else: st.warning("No se pudieron calcular métricas.")
                        except Exception as e: st.error(f"Error al ajustar modelo lineal múltiple: {e}"); st.exception(e)
                    else: st.warning(f"No hay suficientes datos válidos para modelo lineal múltiple.")
            else: st.warning("No hay predictores numéricos disponibles.")

            # --- Regresión no lineal ---
            st.divider()
            st.markdown("### 🔹 Regresión no lineal (Modelos 1 y 2)")
            y_nl = st.selectbox("Objetivo (no lineal)", targets_num_reg, key="nl_y")
            if y_nl:
                 available_preds_nl = [c for c in num_df.columns if c != y_nl and num_df[c].notna().sum() > 20]
                 if available_preds_nl:
                    valid_corr_targets_nl = [p for p in available_preds_nl if p in num_df.columns]; corr_abs_nl = pd.Series(dtype=float)
                    if y_nl in num_df.columns and valid_corr_targets_nl: corr_abs_nl = num_df.corr(numeric_only=True)[y_nl].drop(y_nl, errors='ignore').loc[valid_corr_targets_nl].abs().sort_values(ascending=False)
                    pred_default_nl = corr_abs_nl.index[0] if not corr_abs_nl.empty else available_preds_nl[0]
                    x_nl = st.selectbox("Predictor (X)", available_preds_nl, index=(available_preds_nl.index(pred_default_nl) if pred_default_nl in available_preds_nl else 0), key="nl_x")
                    model_type = st.radio("Seleccionar modelo no lineal", ["Modelo 1: y = a*x² + b*x + c", "Modelo 2: y = C₁*x⁴ + C₂*x²  (Forma: y = (a*x²+b)/c * x²)"], key="nl_model_type", horizontal=True)
                    if x_nl:
                        sub_nl = df_clean[[x_nl, y_nl, 'origen_archivo']].copy()
                        if len(sub_nl[[x_nl, y_nl]].dropna()) >= 10:
                            try:
                                y_vec = sub_nl[y_nl]; modelo_nl = None; Xp_nl = None; X_nl_fit_global = None; x_nl_sq_col = f"{x_nl}_sq"; x_nl_4_col = f"{x_nl}_4"
                                x_num = pd.to_numeric(sub_nl[x_nl], errors='coerce')
                                if not x_num.isna().all():
                                    if x_nl in sub_nl.columns:
                                        if x_nl_sq_col not in sub_nl.columns: sub_nl[x_nl_sq_col] = x_num ** 2
                                        if model_type.startswith("Modelo 2") and x_nl_4_col not in sub_nl.columns: sub_nl[x_nl_4_col] = x_num ** 4
                                        if model_type.startswith("Modelo 1"):
                                            st.markdown("##### Modelo 1: Cuadrático ($y = a*x^2 + b*x + c$)")
                                            if x_nl_sq_col in sub_nl.columns and x_nl in sub_nl.columns:
                                                X_nl_fit_global = sm.add_constant(sub_nl[[x_nl_sq_col, x_nl]], has_constant='add')
                                                min_x_nl, max_x_nl = sub_nl[x_nl].dropna().min(), sub_nl[x_nl].dropna().max()
                                                if pd.notna(min_x_nl) and pd.notna(max_x_nl): xx_nl = np.linspace(min_x_nl, max_x_nl, 200); Xp_nl = pd.DataFrame({"const": 1.0, x_nl_sq_col: xx_nl ** 2, x_nl: xx_nl})
                                                else: Xp_nl = None
                                            else: X_nl_fit_global = None
                                        elif model_type.startswith("Modelo 2"):
                                            st.markdown("##### Modelo 2: Polinomial ($y = C_1*x^4 + C_2*x^2$, sin intercepto)")
                                            if x_nl_4_col in sub_nl.columns and x_nl_sq_col in sub_nl.columns:
                                                X_nl_fit_global = sub_nl[[x_nl_4_col, x_nl_sq_col]]
                                                min_x_nl, max_x_nl = sub_nl[x_nl].dropna().min(), sub_nl[x_nl].dropna().max()
                                                if pd.notna(min_x_nl) and pd.notna(max_x_nl): xx_nl_orig = np.linspace(min_x_nl, max_x_nl, 200); Xp_nl = pd.DataFrame({x_nl_4_col: xx_nl_orig ** 4, x_nl_sq_col: xx_nl_orig ** 2})
                                                else: Xp_nl = None
                                            else: X_nl_fit_global = None
                                    if X_nl_fit_global is not None:
                                        try:
                                            modelo_nl = sm.OLS(y_vec, X_nl_fit_global, missing='drop').fit()
                                            st.markdown("### Resultados del Modelo Global"); st.markdown(f"```text\n{modelo_nl.summary()}\n```")
                                            fig_nl, ax_nl = plt.subplots(); AIRBNB_RED, AIRBNB_TEAL, _ = style_airbnb_plot(ax_nl, fig_nl)
                                            if Xp_nl is not None:
                                                yhat_curve_nl = modelo_nl.predict(Xp_nl)
                                                x_axis_plot = xx_nl_orig if model_type.startswith("Modelo 2") else xx_nl
                                                ax_nl.plot(x_axis_plot, yhat_curve_nl, linewidth=2.5, label="Curva ajustada (Global)", color=AIRBNB_RED, zorder=10)
                                            else: st.warning(f"No se pudo graficar línea para '{x_nl}'.")
                                            sub_plot_nl = sub_nl.dropna(subset=[x_nl, y_nl])
                                            if is_multi_db:
                                                for origin in all_origins: sub_origin_plot = sub_plot_nl[sub_plot_nl['origen_archivo'] == origin]; ax_nl.scatter(sub_origin_plot[x_nl], sub_origin_plot[y_nl], alpha=0.3, label=f"Obs. ({origin})")
                                            else: ax_nl.scatter(sub_plot_nl[x_nl], sub_plot_nl[y_nl], alpha=0.4, label="Observaciones", color=AIRBNB_TEAL)
                                            ax_nl.set_title(f"{y_nl} ~ {x_nl} ({model_type.split(':')[0]})"); ax_nl.set_xlabel(x_nl); ax_nl.set_ylabel(y_nl); ax_nl.legend(); style_airbnb_plot(ax_nl, fig_nl); st.pyplot(fig_nl, use_container_width=True)

                                            st.markdown("#### Comparación de Métricas"); comparison_data = []; p = int(modelo_nl.df_model)
                                            y_hat_global = modelo_nl.fittedvalues; y_eval_global_nl_true = y_vec.loc[y_hat_global.index]
                                            if not y_eval_global_nl_true.empty:
                                                r2_global_std = r2_score(y_eval_global_nl_true, y_hat_global); rmse_global = np.sqrt(mean_squared_error(y_eval_global_nl_true, y_hat_global)); n_global = len(y_eval_global_nl_true); adj_r2_global = calculate_adj_r2(r2_global_std, n_global, p); r2_ols_global = modelo_nl.rsquared
                                                comparison_data.append({"Base de Datos": "Global (Combinada)", "R² (Estándar)": r2_global_std, "R² Ajustado": adj_r2_global, "R² (de OLS)": r2_ols_global, "RMSE": rmse_global, "n": n_global})
                                            if is_multi_db:
                                                for origin in all_origins:
                                                    sub_nl_origin = sub_nl[sub_nl['origen_archivo'] == origin].copy(); x_cols_needed_nl = [x_nl_sq_col, x_nl] if model_type.startswith("Modelo 1") else [x_nl_4_col, x_nl_sq_col]
                                                    if x_nl_sq_col not in sub_nl_origin.columns: sub_nl_origin[x_nl_sq_col] = pd.to_numeric(sub_nl_origin[x_nl], errors='coerce') ** 2
                                                    if model_type.startswith("Modelo 2") and x_nl_4_col not in sub_nl_origin.columns: sub_nl_origin[x_nl_4_col] = pd.to_numeric(sub_nl_origin[x_nl], errors='coerce') ** 4
                                                    sub_origin_cleaned = sub_nl_origin.dropna(subset=[y_nl] + x_cols_needed_nl); n_origin = len(sub_origin_cleaned)
                                                    if n_origin < p + 2: continue
                                                    y_true_origin = sub_origin_cleaned[y_nl]; X_nl_fit_origin = None
                                                    if model_type.startswith("Modelo 1"): X_nl_fit_origin = sm.add_constant(sub_origin_cleaned[x_cols_needed_nl], has_constant='add')
                                                    elif model_type.startswith("Modelo 2"): X_nl_fit_origin = sub_origin_cleaned[x_cols_needed_nl]
                                                    if X_nl_fit_origin is not None and not X_nl_fit_origin.empty:
                                                        y_pred_origin = modelo_nl.predict(X_nl_fit_origin); r2_origin = r2_score(y_true_origin, y_pred_origin); rmse_origin = np.sqrt(mean_squared_error(y_true_origin, y_pred_origin)); adj_r2_origin = calculate_adj_r2(r2_origin, n_origin, p)
                                                        comparison_data.append({"Base de Datos": origin, "R² (Estándar)": r2_origin, "R² Ajustado": adj_r2_origin, "R² (de OLS)": np.nan, "RMSE": rmse_origin, "n": n_origin})
                                            if comparison_data: st.dataframe(pd.DataFrame(comparison_data).set_index('Base de Datos').style.format("{:.3f}", subset=["R² (Estándar)", "R² Ajustado", "R² (de OLS)", "RMSE"], na_rep='N/A'))
                                            else: st.warning("No se pudieron calcular métricas.")
                                        except Exception as e: st.error(f"Error al ajustar OLS no lineal: {e}"); st.exception(e)
                                else: st.warning("No se pudo construir variables para modelo no lineal.")
                            except Exception as e: st.error(f"Error general en sección no lineal: {e}"); st.exception(e)
                        else: st.warning(f"No hay suficientes datos válidos para modelo no lineal.")
                    else: st.warning("Selecciona un predictor (X) válido.")
                 else: st.warning("No hay predictores numéricos disponibles.")
        else: st.warning("Se requieren >= 2 columnas numéricas para modelado.")

    # === INICIO: PESTAÑA 4 (REGRESIÓN LOGÍSTICA) ===
    with tab4:
        st.subheader("Regresión logística binaria")
        friendly_to_tech = {
            title: tech_name
            for tech_name, title in target_titles.items()
            if tech_name in df_bin.columns and df_bin[tech_name].dropna().nunique() > 1
        }
        if not friendly_to_tech: st.error("No se pudo crear/encontrar variable binaria objetivo válida.")
        else:
            friendly_name = st.selectbox("Target binario (variable objetivo)", options=list(friendly_to_tech.keys()), key="logit_target")
            y_logit = friendly_to_tech[friendly_name]
            df_corr_base = df_bin.select_dtypes(include=[np.number]); base_excl = forbidden_map.get(y_logit, set())
            available_predictors = [
                c for c in df_corr_base.columns
                if c not in (derived_bins | extra_exclusions | base_excl | {y_logit})
                   and df_corr_base[c].dropna().nunique() > 1 and pd.api.types.is_numeric_dtype(df_corr_base[c]) and df_corr_base[c].var() > 1e-9
                   and df_corr_base[c].notna().sum() > 20 ]
            default_predictors = []
            if available_predictors:
                try:
                    top_corr_df = top_k_no_other_bins(df_corr_base, y_logit, k=3, base_exclude=base_excl, derived_bins=derived_bins, extra_exclusions=extra_exclusions)
                    if not top_corr_df.empty: valid_defaults = [p for p in top_corr_df['Variable'].tolist() if p in available_predictors]; default_predictors = valid_defaults
                except Exception as e: st.warning(f"No se pudieron calcular correlaciones iniciales: {e}")
                if not default_predictors: default_predictors = available_predictors[:min(3, len(available_predictors))]
            Xs_logit = st.multiselect("Predictores", available_predictors, default=default_predictors, key="logit_predictors")
            c1, c2, c3 = st.columns(3)
            test_size = c1.slider("Proporción de test", 0.1, 0.5, 0.25, 0.05, key="logit_test_size")
            thr = c2.slider("Umbral de clasificación", 0.1, 0.9, 0.5, 0.05, key="logit_threshold")
            use_balanced = c3.checkbox("Reponderar clases (balanced)", value=False, key="logit_balanced")
            class_weight = 'balanced' if use_balanced else None
            if st.button("Entrenar logística", key="logit_train_button") and len(Xs_logit) >= 1:
                sub = df_bin[[y_logit, 'origen_archivo'] + Xs_logit].dropna(subset=[y_logit] + Xs_logit)
                if sub.empty or sub[y_logit].nunique() < 2: st.warning("No hay suficientes datos o target constante tras NAs.")
                else:
                    try:
                        X = sub[Xs_logit].astype(float).values; yv = sub[y_logit].astype(int).values; origenes_array = sub['origen_archivo'].values
                        min_class_count = np.min(np.unique(yv, return_counts=True)[1]) if len(np.unique(yv)) > 1 else 0
                        stratify_option = yv if min_class_count >= 2 else None
                        if stratify_option is None and len(np.unique(yv)) > 1: st.warning("No se pudo estratificar split.")
                        X_train, X_test, y_train, y_test, origin_train, origin_test = train_test_split(X, yv, origenes_array, test_size=test_size, stratify=stratify_option, random_state=42)
                        test_classes = np.unique(y_test); can_calculate_auc = len(test_classes) > 1
                        if not can_calculate_auc: st.warning("Test set solo contiene una clase. ROC AUC no se puede calcular.")
                        sc = StandardScaler().fit(X_train); Xtr = sc.transform(X_train); Xte = sc.transform(X_test)
                        clf = LogisticRegression(max_iter=300, solver="lbfgs", class_weight=class_weight, random_state=42); clf.fit(Xtr, y_train)
                        st.markdown("---"); st.markdown("### Comparación de Métricas del Modelo Global (sobre sets de Test)"); comparison_data = []
                        auc = np.nan; cm_global = None; prec_si = np.nan; acc = np.nan; rec = np.nan
                        if Xte.shape[0] > 0:
                             proba_global = clf.predict_proba(Xte)[:,clf.classes_.tolist().index(1)]; y_pred_global = (proba_global >= thr).astype(int)
                             acc = accuracy_score(y_test, y_pred_global); rec = recall_score(y_test, y_pred_global, zero_division=0)
                             if can_calculate_auc:
                                 try: auc = roc_auc_score(y_test, proba_global)
                                 except ValueError: auc = np.nan
                             cm_global = confusion_matrix(y_test, y_pred_global, labels=clf.classes_)
                             if len(cm_global.ravel()) == 4: tn, fp, fn, tp = cm_global.ravel(); prec_si = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                             comparison_data.append({"Base de Datos": "Global (Test Set)", "Accuracy": acc, "Precisión (SÍ/1)": prec_si, "Recall (SÍ/1)": rec, "ROC AUC": auc, "n": len(y_test)})
                        else: st.warning("Test set global vacío.")
                        if is_multi_db and Xte.shape[0] > 0:
                            df_test = pd.DataFrame(X_test, columns=Xs_logit); df_test['y_true'] = y_test; df_test['proba_1'] = proba_global; df_test['origen_archivo'] = origin_test
                            for origin in all_origins:
                                df_test_origin = df_test[df_test['origen_archivo'] == origin]; origin_test_classes = df_test_origin['y_true'].unique(); can_calc_auc_origin = len(origin_test_classes) > 1
                                if df_test_origin.empty: continue
                                y_test_origin = df_test_origin['y_true']; proba_origin = df_test_origin['proba_1']; y_pred_origin = (proba_origin >= thr).astype(int)
                                acc_orig = accuracy_score(y_test_origin, y_pred_origin); rec_orig = recall_score(y_test_origin, y_pred_origin, zero_division=0); auc_orig = np.nan
                                if can_calc_auc_origin:
                                     try: auc_orig = roc_auc_score(y_test_origin, proba_origin)
                                     except ValueError: auc_orig = np.nan
                                cm_orig = confusion_matrix(y_test_origin, y_pred_origin, labels=clf.classes_); prec_si_orig = np.nan
                                if len(cm_orig.ravel()) == 4: tn_o, fp_o, fn_o, tp_o = cm_orig.ravel(); prec_si_orig = tp_o / (tp_o + fp_o) if (tp_o + fp_o) > 0 else 0.0
                                comparison_data.append({"Base de Datos": origin, "Accuracy": acc_orig, "Precisión (SÍ/1)": prec_si_orig, "Recall (SÍ/1)": rec_orig, "ROC AUC": auc_orig, "n": len(y_test_origin)})
                        if comparison_data: st.dataframe(pd.DataFrame(comparison_data).set_index('Base de Datos').style.format("{:.3f}", na_rep='N/A'))
                        else: st.warning("No se pudieron calcular métricas.")
                        if cm_global is not None:
                            st.markdown("---"); st.markdown("### Detalles del Modelo Global (sobre Test Set Global)"); c_plot1, c_plot2 = st.columns(2)
                            class_labels = clf.classes_; cm_df = pd.DataFrame(cm_global, index=[f"Real {c}" for c in class_labels], columns=[f"Pred {c}" for c in class_labels])
                            c_plot1.markdown(f"**Matriz de confusión {class_labels}**"); c_plot1.dataframe(cm_df)
                            if can_calculate_auc and not np.isnan(auc):
                                try:
                                    fpr, tpr, _ = roc_curve(y_test, proba_global); figR, axR = plt.subplots(figsize=(6,5)); AIRBNB_RED, AIRBNB_TEAL, _ = style_airbnb_plot(axR, figR)
                                    axR.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}", color=AIRBNB_RED, linewidth=2.5); axR.plot([0,1],[0,1],'--', color=AIRBNB_TEAL)
                                    axR.set_xlabel("FPR"); axR.set_ylabel("TPR"); axR.set_title("Curva ROC (Global)"); axR.legend(); style_airbnb_plot(axR, figR); c_plot2.pyplot(figR, use_container_width=True)
                                except Exception as e: c_plot2.info(f"No se pudo graficar ROC. Error: {e}")
                            elif not can_calculate_auc: c_plot2.info("No se puede graficar curva ROC (solo una clase en Test set).")
                            else: c_plot2.info("No se puede graficar curva ROC (AUC no válido).")
                    except Exception as e: st.error(f"Error durante entrenamiento/evaluación logística: {e}"); st.exception(e)
            elif len(Xs_logit) < 1: st.warning("Selecciona al menos 1 predictor.")
    # === FIN: PESTAÑA 4 ===

#####################################################
# Lógica Principal de la Aplicación Streamlit
#####################################################
st.set_page_config(page_title="Análisis Airbnb Multi-Ciudad", page_icon="Logo.png", layout="wide")

st.sidebar.image("Logo.png", width=50) # Logo en la sidebar
st.sidebar.title("Navegación")
PAGINAS = {"Limpieza y Análisis": pagina_limpieza_analisis, "Extracción de Características": pagina_extraccion_caracteristicas, "Modelado (Regresión/Logística)": pagina_modelado}
pagina_seleccionada = st.sidebar.radio("Elige una página", list(PAGINAS.keys()), key="main_page_select")
st.sidebar.markdown("---"); st.sidebar.header("Configuración de Datos ⚙️")
ARCHIVOS_DISPONIBLES = {"Washington DC": "Washington.csv", "Venecia": "Venecia.csv", "Ciudad de México": "CDMX.csv"}
opciones = list(ARCHIVOS_DISPONIBLES.keys()); default_seleccion = ["Venecia"] if "Venecia" in opciones else [opciones[0]] if opciones else []
archivos_seleccionados = st.sidebar.multiselect("1. Selecciona tus bases de datos:", options=opciones, default=default_seleccion, key="main_file_select")
st.sidebar.markdown("2. Selecciona un método de limpieza:")
metodo_limpieza = st.sidebar.radio("Método de Limpieza:", ("Sin Limpieza", "Limpieza Sencilla (Genérica)", "Limpieza Robusta (Específica)"), index=2, key="metodo_limpieza", label_visibility="collapsed")
st.sidebar.info("*Limpieza Sencilla*: Rellena NaNs y elimina outliers numéricos (+-2 Desv. Estándar).\n\n*Limpieza Robusta*: Lógica específica por variable")

# Lógica de Carga y Limpieza
df_original, df = pd.DataFrame(), pd.DataFrame(); cleaning_log = []
if archivos_seleccionados:
    lista_dfs = [load_data(ARCHIVOS_DISPONIBLES[nombre]) for nombre in archivos_seleccionados if ARCHIVOS_DISPONIBLES.get(nombre)]
    if lista_dfs:
        lista_dfs_validas = [df_item for df_item in lista_dfs if not df_item.empty]
        if lista_dfs_validas:
            df_original = pd.concat(lista_dfs_validas, ignore_index=True)
            if not df_original.empty:
                df_preprocesado = preprocesar_dataframe(df_original) # Conversión de moneda aquí
                if metodo_limpieza == "Limpieza Sencilla (Genérica)": df = limpieza_sencilla(df_preprocesado, log_collector=cleaning_log); st.sidebar.success("Procesado con Limpieza Sencilla.")
                elif metodo_limpieza == "Limpieza Robusta (Específica)":
                    try: df = limpieza_robusta(df_preprocesado, log_collector=cleaning_log); st.sidebar.success("Procesado con Limpieza Robusta.")
                    except Exception as e: st.sidebar.error(f"Falló Limp. Robusta: {e}"); df = df_preprocesado
                else: df = df_preprocesado; st.sidebar.info("Mostrando datos pre-procesados.")
            else: st.warning("Df original concatenado vacío."); df = pd.DataFrame()
        else: st.warning("Dfs cargados vacíos/no concatenables."); df = pd.DataFrame()
    else: st.warning("No se pudieron cargar archivos."); df = pd.DataFrame()
else: st.warning("Selecciona >= 1 base de datos."); df = pd.DataFrame()

is_multi_db_main = df['origen_archivo'].nunique() > 1 if not df.empty and 'origen_archivo' in df.columns else False

# Renderizar Página Seleccionada
if pagina_seleccionada == "Limpieza y Análisis":
    st.title("Limpieza y Análisis Exploratorio 🔎"); df_orig_display = df_original if not df_original.empty else pd.DataFrame(); pagina_limpieza_analisis(df_orig_display, df, metodo_limpieza, log_mensajes=cleaning_log)
elif pagina_seleccionada == "Extracción de Características":
    st.title("Dashboard de Características Clave 📊"); pagina_extraccion_caracteristicas(df)
elif pagina_seleccionada == "Modelado (Regresión/Logística)":
    st.title("Modelado Explicativo y Predictivo 📈")
    if is_multi_db_main: st.info("Múltiples bases: Modelos entrenados con datos combinados ('Global') y evaluados individualmente.")
    pagina_modelado(df)
