import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# cargar datos desde el json que descargamos de la api
archivo_local = "historial_nublense_coquimbo.json"

X_lista = []
y_resultados_lista = []
y_goles_lista = []

# abrimos el json si es que existe
if os.path.exists(archivo_local):
    with open(archivo_local, 'r') as f:
        partidos_json = json.load(f)
        
    for partido in partidos_json:
        fixture = partido['fixture']
        equipos = partido['teams']
        goles = partido['goals']
        
        # me salto los partidos que no han terminado o fueron suspendidos
        estado = fixture['status']['short']
        if estado != 'FT' and estado != 'AET' and estado != 'PEN':
            continue
            
        # veo si coquimbo juega de local (id 2330)
        coquimbo_local = 0
        if equipos['home']['id'] == 2330:
            coquimbo_local = 1
            
        # simulo las rachas pq la api gratuita no da el dato de como venian hace años
        racha_coq = np.random.randint(0, 4)
        racha_nub = np.random.randint(0, 4)
        prom_goles_coq = round(np.random.uniform(1.0, 2.0), 2)
        # calcular variable Y: 1 (Gana Coq), 0.5 (Empata), 0 (Pierde Coq)
        if goles['home'] == goles['away']:
            resultado_coq = 0.5
        else:
            if coquimbo_local == 1 and goles['home'] > goles['away']:
                resultado_coq = 1.0
            elif coquimbo_local == 0 and goles['away'] > goles['home']:
                resultado_coq = 1.0
            else:
                resultado_coq = 0.0
                
        # guardo todo en las listas
        X_lista.append([coquimbo_local, racha_coq, racha_nub, prom_goles_coq])
        y_resultados_lista.append(resultado_coq)
        y_goles_lista.append(goles['home'] + goles['away'])

# pasamos las listas a arreglos de numpy
X_historial = np.array(X_lista)
y_historial = np.array(y_resultados_lista)
y_goles_totales = np.array(y_goles_lista)


# ALL IN a ñublense

# modelo de regresión para predecir el partido de ñublense vs coquimbo unido
# DIA del partido: 19/04/2026 a las 12:30 Pm
# toma datos historicos de sus enfrentamientos
modelo = LinearRegression()
modelo.fit(X_historial, y_historial)

# prediccion para los datos historicos
predicciones = modelo.predict(X_historial)
print("R2 Score (que tan bien se ajusta):", round(r2_score(y_historial, predicciones), 2))
print("Error absoluto medio (MAE):", round(mean_absolute_error(y_historial, predicciones), 2))
print("-" * 30)

# evaluar modelo con datos nuevos
# coquimbo es visita (0).
# coquimbo viene con 2 victorias.
# ñublense viene con 1 victoria.
# promedio de goles de coquimbo en el campeonato es 1.25
datos_partido_nuevo = np.array([[0, 2, 1, 1.25]])

prediccion = modelo.predict(datos_partido_nuevo)[0]

# me aseguro que la prob no pase de 1 o baje de 0
probabilidad = max(0, min(prediccion, 1))

print(f"\nPredicción para el partido Ñublense vs Coquimbo Unido (19/04/2026):")
print(f"Probabilidad de que GANE Coquimbo Unido: {probabilidad * 100:.2f}%\n")

if probabilidad > 0.6:
    print("Conclusión: ¡Es muy probable que Coquimbo gane!")
elif probabilidad > 0.4: 
    print("Conclusión: Va a estar reñido, huele a un empate.")
else:
    print("Conclusión: Esta difícil para Coquimbo, ñublense tiene la ventaja.")

# mercado de goles (over/under 2.5)

modelo_goles = LinearRegression()
modelo_goles.fit(X_historial, y_goles_totales)
prediccion_goles = modelo_goles.predict(datos_partido_nuevo)[0]

print("\n--- Apuesta a Cantidad de Goles (Mercado Over/Under) ---")
print(f"Goles totales esperados por el modelo: {prediccion_goles:.2f}")

if prediccion_goles > 2.5:
    print("Sugerencia de apuesta: MÁS DE 2.5 GOLES (Over 2.5) en el partido.")
else:
    print("Sugerencia de apuesta: MENOS DE 2.5 GOLES (Under 2.5) en el partido.")
