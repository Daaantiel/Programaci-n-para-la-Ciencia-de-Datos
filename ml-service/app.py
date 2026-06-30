import pandas as pd
import json
import pickle

from fastapi import FastAPI

#Inicializa la aplicación FastAPI que servirá como el backend de Machine Learning
app = FastAPI(title="Servicio de segmentación de usuarios")

# 1. CARGA DE ARTIFACTOS Y MODELOS AL INICIAR LA API

# Carga la base de datos segmentada final generada por el entrenamiento (train.py)
data = pd.read_csv("data/usuarios_segmentados.csv")

# Carga el modelo predictivo de KMeans entrenado desde ml-service de Python
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))

# Carga el escalador matemático para normalizar datos en futuras predicciones en vivo
scaler = pickle.load(open("/app/models/scaler.pkl", "rb"))

# Carga las métricas del modelo (K óptimo, Silhouette score, etc.)
with open("/app/models/metricas.json") as f:
    metricas = json.load(f)


# 2. ENDPOINTS / RUTAS DE LA API


@app.get("/")
def inicio():
    """
    Ruta raíz (Sanity Check):
    Permite verificar de forma rápida si el contenedor de FastAPI está encendido 
    y respondiendo solicitudes HTTP correctamente en el puerto 8000.
    """
    return {
        "mensaje":
        "Servicio ML funcionando"
    }

@app.get("/dashboard-data")
def dashboard_data():
    """
    Ruta del Dashboard:
    Este endpoint es consumido directamente por la interfaz gráfica de Streamlit.
    Recopila los datos de los usuarios, la ubicación de los centroides de cada cluster 
    y las métricas de rendimiento, transformando todo a formato JSON nativo de la API.
    """
    usuarios = pd.read_csv(
        "/app/data/usuarios_segmentados.csv"
    )

    centroides = pd.read_csv("/app/data/centroides.csv")

    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "metricas": metricas
    }
