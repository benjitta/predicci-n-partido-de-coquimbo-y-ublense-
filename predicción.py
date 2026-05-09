import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

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
