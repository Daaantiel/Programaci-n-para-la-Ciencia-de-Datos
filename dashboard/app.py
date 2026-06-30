import streamlit as st
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt

st.title("Dashboard de Segmentación de Usuarios de Streaming")

# Obtiene los datos para la visualización
respuesta = requests.get(
    "http://ml-service:8000/dashboard-data"
)

payload = respuesta.json()

data = pd.DataFrame(payload["usuarios"])
metricas = payload["metricas"]
centroides = pd.DataFrame(payload["centroides"])

# Muestra las métricas del modelo
st.subheader("Métricas del modelo")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Silhouette Score",
        f"{metricas['silhouette_score']:.3f}"
    )

with col2:
    st.metric(
        "Clusters",
        metricas["n_clusters"]
    )

with col3:
    st.metric(
        "Usuarios",
        metricas["n_usuarios"]
    )

st.subheader("Usuarios segmentados")
st.dataframe(data)
st.subheader("Distribución de segmentos")

st.bar_chart(data["cluster"].value_counts())

# Perfil de cada segmento
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
).round(2)

st.subheader("Perfil de segmentos")
st.dataframe(perfil_segmentos)

# Grafica resultados de PCA
fig, ax = plt.subplots(figsize=(8, 6))

for cluster in sorted(data["cluster"].unique()):
    subset = data[data["cluster"] == cluster]
    ax.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cluster}", alpha=0.7)

ax.set_title("Visualización PCA de los segmentos", fontsize=14, fontweight="bold")
ax.set_xlabel("PC1", fontsize=14, fontweight="bold")
ax.set_ylabel("PC2", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Muestra los cluster usando dos variables
# Carga el modelo
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
# Carga data escalada
scaler = pickle.load(open("models/scaler.pkl", "rb"))
centroides = pd.DataFrame(
    payload["centroides"]
)
st.subheader("Visualización de segmentos usando Gasto Mensual y Horas de Consumo Mensual")
# Escoger dos columnas que se incluirán en el análisis
columna_x = 'gasto_mensual'
columna_y = 'horas_consumo_mensual'
fig, ax = plt.subplots(figsize=(8,6))

scatter = ax.scatter(
    data[columna_x],
    data[columna_y],
    c=data["cluster"],
    alpha=0.7,
    s=40
)

ax.scatter(
    centroides[columna_x],
    centroides[columna_y],
    marker="X",
    s=250,
    edgecolor="black",
    linewidth=2
)

ax.set_xlabel(columna_x, fontsize=14, fontweight="bold")
ax.set_ylabel(columna_y, fontsize=14, fontweight="bold")

ax.set_title(
    f"Clusters según {columna_x} y {columna_y}", fontsize=16, fontweight="bold"
)

ax.grid(True)

st.pyplot(fig)

