import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ALL IN a ñublense

# modelo de regresión para predecir el partido de ñublense vs coquimbo unido
# DIA del partido: 19/04/2025 a las 12:30 Pm
# Tome datos historicos de sus enfrentamiento y como venian en el campeonato

X_historial = np.array([
    [1, 0, 1, 1.25], # 2026-04-03 vs Cobresal (Victoria 3-2) - Local
    [0, 1, 2, 1.25], # 2026-03-14 vs U. de Chile (Derrota 0-1) - Visita
    [1, 0, 1, 1.25], # 2026-03-07 vs Huachipato (Victoria 3-1) - Local
    [0, 0, 1, 1.25], # 2026-02-28 vs D. Concepción (Derrota 0-1) - Visita
    [1, 2, 2, 1.25], # 2026-02-21 vs U. Católica (Derrota 1-3) - Local
    [0, 1, 0, 1.25], # 2026-02-14 vs La Serena (Victoria 1-0) - Visita
    [1, 0, 0, 1.25], # 2026-02-07 vs Otro equipo (Victoria) - Local
    [0, 1, 1, 1.25]  # 2026-01-XX vs Otro equipo (Empate) - Visita
])

y_historial = np.array([1, 0, 1, 0, 0, 1, 1, 0.5])
y_goles_totales = np.array([5, 1, 4, 1, 4, 1, 2, 2])










# Modelo de Goles
modelo_goles = LinearRegression()
modelo_goles.fit(X_historial, y_goles_totales)
prediccion_goles = modelo_goles.predict(datos_partido_nuevo)[0]

print("\n--- Apuesta a Cantidad de Goles (Mercado Over/Under) ---")
print(f"Goles totales esperados por el modelo: {prediccion_goles:.2f}")

if prediccion_goles > 2.5:
    print("Sugerencia de apuesta: MÁS DE 2.5 GOLES (Over 2.5) en el partido.")
else:
    print("Sugerencia de apuesta: MENOS DE 2.5 GOLES (Under 2.5) en el partido.")


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
