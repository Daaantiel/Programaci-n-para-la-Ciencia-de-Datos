# Segmentación de Usuarios de una Plataforma de Streaming con Machine Learning

Este proyecto implementa una solución completa de segmentación de usuarios de streaming utilizando técnicas de aprendizaje no supervisado.

El objetivo es identificar grupos de usuarios con comportamientos y perfiles similares a partir de sus características de consumo y gastos, utilizando el algoritmo **KMeans**.

# La solución integra:
-Un archivo csv con el comportamiento de streaming
-Una segundo csv que se almacena en una base de datos PostGreSQL
-Un proceso de integracion de datos('Train.py)
-Exposicion de los resultados mediante una APIREST con FastApi
-Entrenamiento de un modelo clustering
-Dashboard interactivo desarrollado con StreamLit
-Contenerizacion completa utilizando Docker Compose

# Arquitectura de la solución
    
             CSV Streaming
                      |
                      |
                      v
              +----------------+
              |  Integración   |
              |   de datos     |
              +----------------+
                      |
                      |
        +-------------+-------------+
        |                           |
        v                           v
    PostgreSQL DB                 Dataset integrado
    perfil_usuarios               clientes_segmentados.csv
                      |
                      |
                      v
              +----------------+
              |    KMeans      |
              |  Segmentación  |
              +----------------+

                      |
          +-----------+-----------+
          |                       |

          v                       v

      FastAPI                Streamlit
   Servicio ML              Dashboard


# Tecnologías utilizadas

## Lenguaje
Python 3.11

## Machine Learning
Scikit-learn
KMeans
StandardScaler
PCA
Silhouette Score
KneeLocator

## Datos
Pandas
PostgreSQL
SQLAlchemy

## Backend
- FastAPI
- Uvicorn

## Visualización
- Streamlit
- Matplotlib

## Infraestructura
- Docker
- Docker Compose

# Estructura del proyecto

│   docker-compose.yml
│   README.md
│   
├───dashboard
│       app.py
│       Dockerfile
│       requirements.txt
│       
├───data
│       centroides.csv
│       data_usuarios.csv
│       usuarios_segmentados.csv
│       usuarios_streaming.csv
│       
├───database
│       init.sql
│       perfil_usuarios.csv
│       
├───ml-service
│       app.py
│       Dockerfile
│       requirements.txt
│       train.py
│       
└───models


# Fuentes de datos

## Fuente 1: CSV

Archivo: `usuarios_streaming.csv`

Contiene información relacionada con el comportamiento de los usuarios dentro de la plataforma.

Variables principales:
-id_cliente,horas_consumo_mensual,
-gasto_mensual,
-cantidad_contenidos_vistos,
-sesiones_semana,
-porcentaje_finalizacion,
-tiempo_promedio_sesion_min,
-cantidad_generos_consumidos,
-porcentaje_uso_promociones,
-antiguedad_cliente_meses

## Fuente  2: PostgreSQL
Base de datos: `streaming_db`
Tabla: `perfil_usuarios`

Contiene información complementaria de los usuarios:
id_cliente,
edad,dispositivos_registrados,
porcentaje_uso_app_movil,
cantidad_perfiles_creados,
interacciones_mensuales_soporte,
distancia_promedio_red_km

# Pipeline de Machine Learning
El proceso ejecutado por `train.py` realiza:
1. Lectura del archivo CSV local (`usuarios_streaming.csv`).
2. Conexión a PostgreSQL de forma segura.
3. Extracción de información desde la tabla `perfil_usuarios`.
4. Integración mediante la clave común `id_cliente`.
5. Generación del dataset analítico intermedio `data_usuarios.csv`.
6. Separación de variables y normalización utilizando `StandardScaler`.
7. Evaluación de la inercia y coeficientes de silueta para un rango de $K$ de 2 a 10.
8. Selección automatizada del número óptimo de segmentos usando `KneeLocator`.
9. Reducción de dimensionalidad a 2 componentes con **PCA** para facilitar la visualización.
10. Persistencia del modelo (`modelo_kmeans.pkl`), el escalador (`scaler.pkl`), el objeto PCA (`pca.pkl`), las métricas (`metricas.json`) y las coordenadas reales de los `centroides.csv`.

# Ejecución del proyecto
## Requisitos
Tener instalado:
- Docker
- Docker Compose

## Levantar la solución
Desde la raíz del proyecto:
docker compose up --build

Esto levantará de forma ordenada tres servicios distribuidos:
-ServicioPuertoPostgreSQL (streaming_db) 5432 
-FastAPI (ml-service) 8000 
-Streamlit (dashboard) 8501

# Acceso a los servicios

## API Machine Learning

Abrir: http://localhost:8000

Respuesta esperada:
{
  "mensaje": "Servicio ML funcionando"
}

## Dashboard (Streamlit)

Abrir: http://localhost:8501

El dashboard permite de forma dinámica:

* Visualizar métricas del modelo (Silhouette Score, clusters óptimos, total de usuarios).

* Explorar el DataFrame de usuarios segmentados.

* Analizar la distribución de los segmentos mediante gráficos de barras de conteo.

* Estudiar el perfil promedio de cada grupo con agregaciones detalladas.

* Visualizar la dispersión espacial de los datos mediante las componentes PCA.

* Interpretar geométricamente los clusters y la posición de sus centroides usando las variables de negocio directas: gasto_mensual  y horas_consumo_mensual.

# Endpoint del Dashboard
## Obtener datos para el dashboard

GET /dashboard-data

Retorna:

* Usuarios con su cluster asignado.
* Centroides.
* Métricas del modelo.


## Segmento Perfil:
0 Clientes Moderados (Consumo moderado y gasto medio)
1 Clientes de Bajo Costo (Bajo gasto, uso eficiente de la app)
2 Clientes Premium (Máximo gasto y máximo enganche)

El perfil se obtiene analizando:
- promedio de gasto.
- horas de consumo mensual.
- edad promedio.
- comportamiento digital.
- características comerciales.

# Detener los servicios
docker compose down

Eliminar también los volúmenes:

docker compose down -v

# Cambios en el código
Reconstruir las imágenes:
docker compose build --no-cache

Levantar nuevamente los servicios:
docker compose up

# Verificar el contenido de la base de datos
docker exec -it streaming_db psql -U admin -d streaming_db

Ver las tablas:
\dt

# Objetivo del proyecto
Construir una solución analítica completa que permita transformar datos de comportamiento y perfiles de usuarios de streaming, provenientes de múltiples fuentes, en información accionable para la toma de decisiones comerciales mediante técnicas de Machine Learning no supervisado.
