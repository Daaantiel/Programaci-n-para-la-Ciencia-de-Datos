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

os.makedirs("models", exist_ok=True)

# Archivo CSV
usuarios= pd.read_csv('data/usuarios_streaming.csv')

# Fuente desde la BD
engine = create_engine("postgresql://admin:admin@postgres:5432/streaming_db")

perfil= pd.read_sql(
"""
select *
from perfil_usuarios
""",
engine
)

# Integración
data = usuarios.merge(perfil, on="id_cliente")

# Guarda el archivo con la data integrada
data.to_csv("data/data_usuarios.csv", index=False)

# Variables del modelo: quitamos el ID para dejar solo datos numéricos
X = data.drop(columns=["id_cliente"])

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouettes = []
for k in range(2,11):
    modelo = KMeans(n_clusters=k, random_state=29, n_init=10)
    modelo.fit(X_scaled)

    inertias.append(modelo.inertia_)
    silhouettes.append(silhouette_score(X_scaled, modelo.labels_))

kl = KneeLocator(
    range(2,11),
    inertias,
    curve='convex',
    direction='decreasing'
)

# Modelo
k_optimo = kl.elbow
kmeans = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)
# Predicciones
clusters = kmeans.fit_predict(X_scaled)
data["cluster"] = clusters

print("Modelo de segmentación creado!!!")

pca = PCA(n_components=2)

componentes = pca.fit_transform(X_scaled)

data["pc1"] = componentes[:, 0]
data["pc2"] = componentes[:, 1]

# Guarda data con los cluster y dos componentes principales
data.to_csv("data/usuarios_segmentados.csv", index=False)

# Guarda las métricas 
metricas = {
    "k_optimo": int(k_optimo),
    "silhouette_score": silhouette_score(X_scaled, data["cluster"]),
    "n_usuarios": int(len(data)),
    "n_clusters": int(k_optimo),
    "varianza_pca": float(
        pca.explained_variance_ratio_.sum()
    )
}

with open("models/metricas.json", "w") as f:
    json.dump(metricas, f, indent=4)

# Guarda los cenroides
centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)

centroides_df = pd.DataFrame(
    centroides_original,
    columns=X.columns
)

centroides_df.to_csv("data/centroides.csv", index=False)

# Guardar modelo y data escalada
pickle.dump(kmeans, open("models/modelo_kmeans.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(pca, open("models/pca.pkl", "wb"))

print("Modelo guardado")