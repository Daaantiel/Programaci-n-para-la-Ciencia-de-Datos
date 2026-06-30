import streamlit as st
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt

# Configura el título principal que se desplegará arriba de la página web
st.title("Dashboard de Segmentación de Usuarios de Streaming")

# 1. CONSUMO DE DATOS DESDE LA API DE MACHINE LEARNING (FASTAPI)

#Realiza una petición HTTP GET al contenedor de ML para extraer los resultados
respuesta = requests.get(
    "http://ml-service:8000/dashboard-data"
)

# Transforma la respuesta de la API a un formato JSON estructurado en Python
payload = respuesta.json()

# Convierte las diferentes ramas del JSON en DataFrames de Pandas listos para procesar
data = pd.DataFrame(payload["usuarios"])
metricas = payload["metricas"]
centroides = pd.DataFrame(payload["centroides"])

# Muestra las métricas del modelo
st.subheader("Métricas del modelo")

# Divide la pantalla de Streamlit en 3 columnas proporcionales
col1, col2, col3 = st.columns(3)

with col1:
    # Muestra el indicador de la Silueta (cuán bien separados están los grupos) con 3 decimales
    st.metric(
        "Silhouette Score",
        f"{metricas['silhouette_score']:.3f}"
    )

with col2:
    # Muestra la cantidad final de clusters óptimos encontrados en el entrenamiento
    st.metric(
        "Clusters",
        metricas["n_clusters"]
    )

with col3:
    # Muestra el total de registros de usuarios cargados y procesados
    st.metric(
        "Usuarios",
        metricas["n_usuarios"]
    )

# 3. VISUALIZACIÓN DE TABLAS Y GRÁFICOS INICIALES

st.subheader("Usuarios segmentados")
# Despliega la tabla interactiva con toda la data integrada y etiquetada
st.dataframe(data)
st.subheader("Distribución de segmentos")

# Genera una gráfica de barras nativa para ver cuántos usuarios cayeron en cada grupo
st.bar_chart(data["cluster"].value_counts())

# 4. PERFILAMIENTO SOCIODEMOGRÁFICO Y DE CONSUMO (AGRUPACIÓN)
perfil_segmentos = data.groupby("cluster").agg(
    usuarios=("id_cliente", "count"),
    horas_consumidas=("horas_consumo_mensual", "mean"),
    gasto=("gasto_mensual", "mean"),
    contenidos=("cantidad_contenidos_vistos", "mean"),
    sesiones=("sesiones_semana", "mean"),
    finalizacion=("porcentaje_finalizacion", "mean"),
    tiempo_sesion=("tiempo_promedio_sesion_min", "mean"),
    generos=("cantidad_generos_consumidos", "mean"),
    promociones=("porcentaje_uso_promociones", "mean"),
    antiguedad=("antiguedad_cliente_meses", "mean"),
    edad=("edad", "mean"),
    dispositivos=("dispositivos_registrados", "mean"),
    perfiles=("cantidad_perfiles_creados", "mean")
).round(2) # Redondea los resultados a 2 decimales para facilitar su lectura


st.subheader("Perfil de segmentos")
# Despliega la matriz promedio que define los hábitos de cada grupo de clientes
st.dataframe(perfil_segmentos)

# 5. GRÁFICA PCA

fig, ax = plt.subplots(figsize=(8, 6))

# Itera sobre cada cluster único para pintarlo de un color diferente en el plano cartesiano
for cluster in sorted(data["cluster"].unique()):
    subset = data[data["cluster"] == cluster]
    ax.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster}", alpha=0.7)

ax.set_title("Visualización PCA de los segmentos", fontsize=14, fontweight="bold")
ax.set_xlabel("PC1", fontsize=14, fontweight="bold")
ax.set_ylabel("PC2", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True)

# Renderiza la figura de Matplotlib directamente en la aplicación web de Streamlit
st.pyplot(fig)

# 6. GRÁFICA 2 INTERPRETACION DIRECTA DE VARIABLES

# Muestra los cluster usando dos variables
# Carga el modelo en el volumen común
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
# Carga data escalada
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Recupera las coordenadas des-escaladas de los centros de cada grupo
centroides = pd.DataFrame(
    payload["centroides"]
)

st.subheader("Visualización de segmentos usando Gasto Mensual y Horas de Consumo Mensual")

# Escoger dos columnas que se incluirán en el análisis
columna_x = 'gasto_mensual'
columna_y = 'horas_consumo_mensual'
fig, ax = plt.subplots(figsize=(8,6))

# Dibuja la nube de puntos con los usuarios coloreados según su cluster numérico
scatter = ax.scatter(
    data[columna_x],
    data[columna_y],
    c=data["cluster"],
    alpha=0.7,
    s=40
)

# Superpone marcas en forma de "X" gigante para señalar dónde están los centroides geométricos
ax.scatter(
    centroides[columna_x],
    centroides[columna_y],
    marker="X",
    s=250,
    edgecolor="black",
    linewidth=2
)

# Configura las etiquetas del gráfico con formato legible
ax.set_xlabel(columna_x, fontsize=14, fontweight="bold")
ax.set_ylabel(columna_y, fontsize=14, fontweight="bold")

ax.set_title(
    f"Clusters según {columna_x} y {columna_y}", fontsize=16, fontweight="bold"
)

ax.grid(True)

# Renderiza el gráfico bidimensional final en la pantalla
st.pyplot(fig)

