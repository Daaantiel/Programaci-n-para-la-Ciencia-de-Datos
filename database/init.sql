CREATE TABLE perfil_usuarios (
    id_cliente INT PRIMARY KEY,
    edad INT,
    dispositivos_registrados INT,
    porcentaje_uso_app_movil NUMERIC(5,2),
    cantidad_perfiles_creados INT,
    interacciones_mensuales_soporte INT,
    distancia_promedio_red_km NUMERIC(5,2)
);

COPY perfil_usuarios (
    id_cliente,
    edad,
    dispositivos_registrados,
    porcentaje_uso_app_movil,
    cantidad_perfiles_creados,
    interacciones_mensuales_soporte,
    distancia_promedio_red_km
)
FROM '/database/perfil_usuarios.csv'
DELIMITER ','
CSV HEADER;

