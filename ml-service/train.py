import pandas as pd
import json
import pickle
import os

from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA

# 1. PREPARACIÓN DEL ENTORNO Y CARGA DE DATOS

# Crea la carpeta donde se guardarán los archivos del modelo si no existe
os.makedirs("models", exist_ok=True)

# Carga los datos de comportamiento de los usuarios desde el archivo CSV local
usuarios= pd.read_csv('data/usuarios_streaming.csv')

# Establece la conexión con la base de datos PostgreSQL en el contenedor
engine = create_engine("postgresql://admin:admin@postgres:5432/streaming_db")

# Descarga la tabla de perfiles demográficos desde la base de datos
perfil= pd.read_sql(
"""
select *
from perfil_usuarios
""",
engine
)

# 2. INTEGRACIÓN Y LIMPIEZA DE DATOS

# Cruza ambas fuentes de datos usando la clave común 'id_cliente'
data = usuarios.merge(perfil, on="id_cliente")

# Guarda el archivo con la data integrada
data.to_csv("data/data_usuarios.csv", index=False)

# Separa las variables numéricas para el modelo, eliminando el identificador 
X = data.drop(columns=["id_cliente"])

# 3. PREPROCESAMIENTO (ESCALAMIENTO)

# Inicializa el escalador para normalizar los datos
scaler = StandardScaler()

# Transforma las variables para que tengan la misma escala y no sesguen el modelo
X_scaled = scaler.fit_transform(X)

# 4. OPTIMIZACIÓN DE CLUSTERS (MÉTODO DEL CODO)

inertias = []
silhouettes = []

# Evalúa el rendimiento del algoritmo probando desde 2 hasta 10 clusters
for k in range(2,11):
    modelo = KMeans(n_clusters=k, random_state=29, n_init=10)
    modelo.fit(X_scaled)

# Registra la inercia (distancia interna) y la silueta (separación) de cada K
    inertias.append(modelo.inertia_)
    silhouettes.append(silhouette_score(X_scaled, modelo.labels_))

# Identifica matemáticamente el "punto de codo" óptimo usando KneeLocator
kl = KneeLocator(
    range(2,11),
    inertias,
    curve='convex',
    direction='decreasing'
)

# 5. ENTRENAMIENTO DEL MODELO 

# Extrae el número óptimo de clusters calculado por el algoritmo
k_optimo = kl.elbow
# Inicializa y entrena el KMeans final con el K óptimo
kmeans = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)

# Asigna la etiqueta del cluster correspondiente a cada usuario en el dataset
clusters = kmeans.fit_predict(X_scaled)
data["cluster"] = clusters

print("Modelo de segmentación creado!!!")

# 6. REDUCCIÓN DE DIMENSIONALIDAD (PCA)

# Configura PCA para reducir las variables a 2 componentes principales
pca = PCA(n_components=2)

componentes = pca.fit_transform(X_scaled)

# Guarda las coordenadas PCA para que el Dashboard pueda crear el gráfico 2D
data["pc1"] = componentes[:, 0]
data["pc2"] = componentes[:, 1]

# Guarda el dataset final con las etiquetas de cluster y componentes PCA
data.to_csv("data/clientes_segmentados.csv", index=False)

# 7. GUARDADO DE MÉTRICAS, CENTROIDES Y ARTIFACTOS

# Compila las métricas de rendimiento del modelo para el reporte del Dashboard 
metricas = {
    "k_optimo": int(k_optimo),
    "silhouette_score": silhouette_score(X_scaled, data["cluster"]),
    "n_usuarios": int(len(data)),
    "n_clusters": int(k_optimo),
    "varianza_pca": float(
        pca.explained_variance_ratio_.sum()
    )
}

# Guarda las métricas en un archivo JSON estructurado
with open("models/metricas.json", "w") as f:
    json.dump(metricas, f, indent=4)

# Revierte el escalamiento de los centroides para interpretarlos en sus unidades originales
centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)

# Convierte los centroides en un DataFrame y los guarda en un CSV
centroides_df = pd.DataFrame(
    centroides_original,
    columns=X.columns
)

centroides_df.to_csv("data/centroides.csv", index=False)

# Exporta los modelos entrenados usando Pickle para producción
pickle.dump(kmeans, open("models/modelo_kmeans.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(pca, open("models/pca.pkl", "wb"))

print("Modelo guardado")