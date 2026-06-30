import pandas as pd
import json
import pickle

from fastapi import FastAPI

app = FastAPI(title="Servicio de segmentación de usuarios")

# Carga la data que fue guardada en el entrenamiento
data = pd.read_csv("data/usuarios_segmentados.csv")

# Carga el modelo
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
# Carga data escalada
scaler = pickle.load(open("/app/models/scaler.pkl", "rb"))

# Carga las métricas
with open("/app/models/metricas.json") as f:
    metricas = json.load(f)

@app.get("/")
def inicio():
    return {
        "mensaje":
        "Servicio ML funcionando"
    }

@app.get("/dashboard-data")
def dashboard_data():
    usuarios = pd.read_csv(
        "/app/data/usuarios_segmentados.csv"
    )

    centroides = pd.read_csv("/app/data/centroides.csv")

    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "metricas": metricas
    }
