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

# Modelo de Probabilidad
modelo = LinearRegression()
modelo.fit(X_historial, y_historial)

# Predicción para los datos históricos
predicciones = modelo.predict(X_historial)
print("R2 Score (que tan bien se ajusta):", round(r2_score(y_historial, predicciones), 2))
print("Error absoluto medio (MAE):", round(mean_absolute_error(y_historial, predicciones), 2))
print("-" * 30)

# Evaluar modelo con datos nuevos
datos_partido_nuevo = np.array([[0, 2, 1, 1.25]])
prediccion = modelo.predict(datos_partido_nuevo)[0]
probabilidad = max(0, min(prediccion, 1))

print(f"\nPredicción para el partido Ñublense vs Coquimbo Unido (19/04/2025):")
print(f"Probabilidad de que GANE Coquimbo Unido: {probabilidad * 100:.2f}%\n")

if probabilidad > 0.6:
    print("Conclusión: ¡Es muy probable que Coquimbo gane!")
elif probabilidad > 0.4: 
    print("Conclusión: Va a estar reñido, huele a un empate.")
else:
    print("Conclusión: Esta difícil la para Coquimbo ñublense tiene la ventaja.")

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

# 1. Gráfico de Probabilidad (Gráfico de Torta)
etiquetas = ['Probabilidad Coquimbo', 'Ventaja Ñublense / Empate']
tamanos = [probabilidad * 100, (1 - probabilidad) * 100]
colores = ['#ffe100', '#cc0000'] # Amarillo para Coquimbo, Rojo para Ñublense

plt.figure(figsize=(6, 6))
plt.pie(tamanos, labels=etiquetas, colors=colores, autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Predicción de Victoria: Ñublense vs Coquimbo')
plt.savefig('grafico_probabilidad.png')
plt.show()

# 2. Gráfico de Goles (Gráfico de Barras)
plt.figure(figsize=(6, 4))
plt.bar(['Goles Esperados del Modelo'], [prediccion_goles], color='#2b8cbe')
plt.axhline(y=2.5, color='red', linestyle='--', linewidth=2, label='Línea de Apuesta (2.5)')
plt.ylim(0, max(4, prediccion_goles + 1))
plt.title('Mercado Over/Under: Goles Totales')
plt.ylabel('Cantidad de Goles')
plt.legend()
plt.savefig('grafico_goles.png')
plt.show()
