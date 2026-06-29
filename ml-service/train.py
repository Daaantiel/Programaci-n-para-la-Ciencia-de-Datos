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

engine = create_engine("postgresql://admin:admin@postgres:5432/streaming_db")

perfil= pd.read_sql(
"""
select *
from perfil_usuarios
""",
engine
)

#integracion

data= usuarios.merge(perfil, on="id_cliente")

# Guarda el archivo con la data integrada
data.to_csv("data/usuarios_streaming.csv", index=False)

# Variables del modelo
X= data.drop(columns=["id_cliente"])

# Escalamiento
scaler= StandardScaler()
X_scaled= scaler.fit_transform(X)

inertias= []
shilhouette_scores= []
for k in range(2, 11):
    modelo_kmeans= KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo_kmeans.fit(X_scaled)
    inertias.append(modelo_kmeans.inertia_)
    shilhouette_scores.append(silhouette_score(X_scaled, modelo_kmeans.labels_))
    
kl= KneeLocator(range(2, 11), inertias, curve="convex", direction="decreasing")

#Modelo
K_optimo= kl.elbow
K_means= KMeans(n_clusters=K_optimo, random_state=42, n_init=10)

#predicciones del modelo 
clusters= K_means.fit_predict(X_scaled)
data["cluster"]= clusters

print("Modelo de segmentación entrenado con éxito. Número de clusters: ", K_optimo)

pca= PCA(n_components=2)

components= pca.fit_transform(X_scaled)

data["pca1"]= components[:, 0]
data["pca2"]= components[:, 1]

# Guarda data con los cluster y dos componentes principales
data.to_csv("data/usuarios_streaming_cluster.csv", index=False)

# Guarda las métricas 

metricas= {
    "n_clusters": K_optimo,
    "inertia": K_means.inertia_,
    "silhouette_score": silhouette_score(X_scaled, K_means.labels_)
    
}

with open("models/metricas.json", "w") as f:
    json.dump(metricas, f)

#Guardado de centroides
centroides= scaler.inverse_transform(K_means.cluster_centers_)

centroides_df= pd.DataFrame(centroides, columns=X.columns)
centroides_df.to_csv("data/centroides.csv", index=False)

#Guardado del modelo y escalado
pickle.dump(K_means, open("models/kmeans_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb")) 
pickle.dump(pca, open("models/pca.pkl", "wb"))
