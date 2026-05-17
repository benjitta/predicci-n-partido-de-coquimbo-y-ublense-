import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Datos
#[local, goles_favor, goles_contra, promedio_goles_campeonato, victorias3, empates3, derrotas3, promedio_goles_ultimos5]
X_nublense = np.array([
    [1,1,0,1.13,2,1,0,2.00],
    [0,0,0,1.13,2,1,0,1.75],
    [0,2,0,1.33,2,1,0,2.00],
    [1,1,0,1.11,1,2,0,2.25],
    [0,1,1,1.13,1,1,1,2.00],
    [1,2,2,1.14,2,0,1,1.80],
    [0,2,0,1.00,2,0,1,2.00],
    [1,1,2,1.00,2,1,0,2.00],
    [0,1,0,1.00,2,1,0,2.00],
    [1,1,0,1.00,1,1,0,2.40],
    [1,1,1,1.00,0,1,0,2.20],
    [0,1,1,1.00,0,1,0,2.20],
    [1,1,0,1.16,1,1,1,2.00],
    [0,1,0,1.22,1,1,1,1.60],
    [0,0,1,1.17,0,1,2,1.40],
    [0,0,2,1.23,0,0,3,1.60],
    [1,1,0,1.29,1,0,2,2.00],
    [0,0,1,1.30,1,1,1,2.40],
    [1,2,2,1.37,1,2,0,2.20],
    [1,2,0,1.33,2,1,0,2.20],
    [0,2,1,1.29,2,1,0,2.00],
    [1,1,1,1.25,2,1,0,2.40],
    [0,0,2,1.27,1,1,1,2.00],
    [1,1,0,1.36,2,1,0,1.80],
    [0,1,1,1.38,2,1,0,1.80],
    [1,0,1,1.42,2,1,0,2.00],
    [0,1,0,1.55,2,0,1,1.80],
    [1,2,1,1.60,1,0,2,2.60],
    [0,2,3,1.75,1,1,1,2.20],
    [1,2,2,2.25,2,1,0,0.00]
])

X_coquimbo = np.array([
    [0,2,1,2.0,0,0,0,0.00],
    [1,2,1,2.0,1,0,0,3.00],
    [0,1,1,1.6,2,0,0,3.00],
    [1,4,2,2.2,2,1,0,2.67],
    [1,3,2,2.4,2,1,0,3.50],
    [1,0,0,2.0,2,1,0,4.00],
    [0,0,1,1.7,2,1,0,3.20],
    [1,3,1,1.8,1,1,1,2.80],
    [0,1,0,1.7,1,1,1,3.20],
    [0,1,3,1.7,2,0,1,2.20],
    [1,0,1,1.5,2,0,1,2.00],
    [0,3,1,1.6,1,0,2,2.00],
    [1,0,1,1.5,1,0,2,2.40],
    [0,1,1,1.5,1,0,2,2.20],
    [0,1,0,1.4,0,1,2,2.40],
    [1,2,1,1.5,1,1,1,1.80],
    [1,3,2,1.6,2,1,0,2.20],
    [1,1,1,1.5,3,0,0,3.00],
    [0,2,0,1.6,2,1,0,2.60],
    [1,1,1,1.5,2,1,0,3.00],
    [0,2,1,1.6,1,2,0,2.60],
    [0,0,3,1.5,2,1,0,2.60],
    [0,1,3,1.4,1,1,1,2.20],
    [1,2,1,1.5,0,1,2,2.80],
    [1,2,2,1.5,1,0,2,2.60],
    [1,1,0,1.5,1,1,1,3.40],
    [1,3,0,1.55,2,1,0,3.00]
])

datos_crudos = np.vstack((X_nublense, X_coquimbo))
columnas = ['local', 'goles_favor', 'goles_contra', 'promedio_goles_campeonato', 'victorias3', 'empates3', 'derrotas3', 'promedio_goles_ultimos5']
df = pd.DataFrame(datos_crudos, columns=columnas)

# Generar variables objetivo (Targets)
# resultado: 1 si gana, 0.5 si empata, 0 si pierde
df['resultado'] = np.where(df['goles_favor'] > df['goles_contra'], 1,
                           np.where(df['goles_favor'] == df['goles_contra'], 0.5, 0))

# goles_totales: goles_favor + goles_contra
df['goles_totales'] = df['goles_favor'] + df['goles_contra']

# Filtrar variables irrelevantes/problemáticas para X (evitar data leakage)
# goles_favor y goles_contra ocurren durante el partido, no se pueden usar para predecir antes del mismo.
features = ['local', 'promedio_goles_campeonato', 'victorias3', 'empates3', 'derrotas3', 'promedio_goles_ultimos5']
X = df[features]
y_resultado = df['resultado']
y_goles = df['goles_totales']

# =============== MODELOS ===============
modelo_prob = LinearRegression()
modelo_prob.fit(X, y_resultado)
predicciones_prob = modelo_prob.predict(X)

modelo_goles = LinearRegression()
modelo_goles.fit(X, y_goles)
predicciones_goles = modelo_goles.predict(X)

print("--- Evaluación Modelo de Probabilidad ---")
print("R2 Score:", round(r2_score(y_resultado, predicciones_prob), 2))
print("MAE:", round(mean_absolute_error(y_resultado, predicciones_prob), 2))

# Predicción para el partido Ñublense vs Coquimbo
# Datos recientes de Coquimbo (Visita = 0, y promedios de su última fecha)
datos_partido_nuevo = pd.DataFrame([[0, 1.55, 2, 1, 0, 3.00]], columns=features)
pred_prob = modelo_prob.predict(datos_partido_nuevo)[0]
prob_coquimbo = max(0, min(pred_prob, 1))

print(f"\nPredicción para el partido Ñublense vs Coquimbo Unido:")
print(f"Probabilidad de que GANE Coquimbo Unido: {prob_coquimbo * 100:.2f}%\n")
if prob_coquimbo > 0.6:
    print("Conclusión: ¡Es muy probable que Coquimbo gane!")
elif prob_coquimbo > 0.4: 
    print("Conclusión: Va a estar reñido, huele a un empate.")
else:
    print("Conclusión: Esta difícil para Coquimbo, Ñublense tiene la ventaja.")

pred_goles = modelo_goles.predict(datos_partido_nuevo)[0]
print("\n--- Apuesta a Cantidad de Goles (Mercado Over/Under) ---")
print(f"Goles totales esperados por el modelo: {pred_goles:.2f}")

if pred_goles > 2.5:
    print("Sugerencia de apuesta: MÁS DE 2.5 GOLES (Over 2.5) en el partido.")
else:
    print("Sugerencia de apuesta: MENOS DE 2.5 GOLES (Under 2.5) en el partido.")

# ================= GRÁFICOS =================

sns.set_theme(style="whitegrid")

# 1. Matriz de Correlación (Heatmap / Mapa de Calor)
plt.figure(figsize=(10, 8))
# Usamos df completo para ver correlaciones incluso con goles_favor y goles_contra
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación de Variables')
plt.tight_layout()
plt.savefig('matriz_correlacion.png')
plt.show()

# 2. Gráficos de Dispersión (Scatter plots) - Variables vs Target
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x='promedio_goles_campeonato', y='resultado', data=df, color='blue', alpha=0.6)
plt.title('Promedio Goles Campeonato vs Resultado')
plt.subplot(1, 2, 2)
sns.scatterplot(x='promedio_goles_ultimos5', y='resultado', data=df, color='green', alpha=0.6)
plt.title('Promedio Goles Últimos 5 vs Resultado')
plt.tight_layout()
plt.savefig('graficos_dispersion.png')
plt.show()

# 3. Gráfico de Residuos (Probabilidad)
residuos_prob = y_resultado - predicciones_prob
plt.figure(figsize=(8, 5))
sns.scatterplot(x=predicciones_prob, y=residuos_prob, color='purple', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Gráfico de Residuos (Modelo Resultado)')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.tight_layout()
plt.savefig('grafico_residuos.png')
plt.show()

# 4. Gráfico de Dispersión (Predicciones vs Reales - Goles)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_goles, y=predicciones_goles, color='orange', alpha=0.7)
plt.plot([y_goles.min(), y_goles.max()], [y_goles.min(), y_goles.max()], 'r--', linewidth=2)
plt.title('Dispersión: Predicciones vs Reales (Goles Totales)')
plt.xlabel('Goles Reales')
plt.ylabel('Predicción de Goles')
plt.tight_layout()
plt.savefig('dispersion_pred_vs_real.png')
plt.show()

# 5. Gráfico de Cajas
plt.figure(figsize=(12, 6))
sns.boxplot(data=X, palette="Set2")
plt.title('Gráfico de Cajas de las Variables Predictoras')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('grafico_cajas.png')
plt.show()

# 6. Histograma
plt.figure(figsize=(8, 5))
sns.histplot(df['promedio_goles_campeonato'], bins=12, kde=True, color='teal')
plt.title('Histograma - Promedio Goles Campeonato')
plt.xlabel('Promedio Goles')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('histograma_variable.png')
plt.show()

# 7. Histograma de Residuos
plt.figure(figsize=(8, 5))
sns.histplot(residuos_prob, bins=12, kde=True, color='brown')
plt.title('Histograma de Residuos (Modelo Resultado)')
plt.xlabel('Residuo')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('histograma_residuos.png')
plt.show()

# 8. Gráfico de Probabilidad (Gráfico de Torta)
etiquetas = ['Probabilidad Coquimbo', 'Ventaja Ñublense / Empate']
tamanos = [prob_coquimbo * 100, (1 - prob_coquimbo) * 100]
colores = ['#ffe100', '#cc0000'] # Amarillo para Coquimbo, Rojo para Ñublense

plt.figure(figsize=(6, 6))
plt.pie(tamanos, labels=etiquetas, colors=colores, autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Predicción de Victoria: Ñublense vs Coquimbo')
plt.savefig('grafico_probabilidad.png')
plt.show()

# 9. Gráfico de Goles (Gráfico de Barras)
plt.figure(figsize=(6, 4))
plt.bar(['Goles Esperados del Modelo'], [pred_goles], color='#2b8cbe')
plt.axhline(y=2.5, color='red', linestyle='--', linewidth=2, label='Línea de Apuesta (2.5)')
plt.ylim(0, max(4, pred_goles + 1))
plt.title('Mercado Over/Under: Goles Totales')
plt.ylabel('Cantidad de Goles')
plt.legend()
plt.savefig('grafico_goles.png')
plt.show()
