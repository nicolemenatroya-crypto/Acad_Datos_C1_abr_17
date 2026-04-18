# Objetivo: Entender cómo usar vectores y matrices para construir un score de negocio
import numpy as np
import pandas as pd

# Vamos a simular un problema de scoring de clientes
# Variables: edad, ingresos, deuda, antigüedad
data = pd.DataFrame({
    "edad": [25, 40, 35, 50, 28],
    "ingresos": [1500, 3000, 2500, 4000, 1800],
    "deuda": [300, 1200, 800, 1500, 400],
    "antiguedad": [2, 10, 5, 15, 3]
})

print("Datos de clientes:")
print(data) 

# Un cliente como vector
cliente_1 = data.iloc[0].values

# Interpretación
print("\nVector del cliente 1:")
print(cliente_1)
print("Edad:", cliente_1[0])
print("Ingresos:", cliente_1[1])
print("Deuda:", cliente_1[2])
print("Antigüedad:", cliente_1[3])

# Dataset completo como matriz
X = data.values
print("\nMatriz de datos:")
print(X)

# Dimensiones (filas, columnas)
print("\nDimensiones de la matriz:")
print(X.shape)

# Definimos pesos para cada variable según su importancia en el scoring
# Edad: impacto positivo leve, 
# Ingresos: impacto positivo fuerte, 
# Deuda: impacto negativo, 
# Antigüedad: impacto positivo fuerte
pesos = np.array([0.1, 0.0005, -0.002, 0.2])
print("\nPesos para el scoring:")
print(pesos)

# Producto punto

# Cálculo manual del score para el cliente 1
score_manual = (cliente_1[0] * pesos[0] +
                cliente_1[1] * pesos[1] +
                cliente_1[2] * pesos[2] +
                cliente_1[3] * pesos[3])
print("\nScore calculado manualmente para el cliente 1:")
print(score_manual)

# Score de un cliente
score_cliente_1 = np.dot(cliente_1, pesos)
print("\nScore del cliente 1:")
print(score_cliente_1)


# Aplicar Score para todos los clientes
scores = np.dot(X, pesos)
print("\nScores de todos los clientes:")
print(scores)

# Agregar al dataset
data["score"] = scores
print("\nDataset aumentado con scores:")
print(data)

# Toma de decisiones con un umbral
umbral = 5
data["decision"] = data["score"].apply(lambda x: "Aprobar" if x >= umbral else "Rechazar")
print("\nDataset con decisiones:")
print(data)

# Ordenar por score
data_sorted = data.sort_values(by="score", ascending=False)
print("\nClientes ordenados por score:")
print(data_sorted)

# Interpretación
# Cliente con mayor score
print("\nCliente con mayor score:")
print(data_sorted.iloc[0])

# ¿Por qué este cliente tiene mayor score?
# ¿Qué variable influyó más?

# Sensibilidad del score a los pesos
# Cambiar pesos

pesos_alt = np.array([0.05, 0.001, -0.003, 0.1])
scores_alt = np.dot(X, pesos_alt)

data["score_alt"] = scores_alt
print("\nDataset con scores alternativos:")
print(data)

# Nuevo ranking
data.sort_values(by="score_alt", ascending=False)
print("\nClientes ordenados por score alternativo:")
print(data.sort_values(by="score_alt", ascending=False))

# Comparación
print("\nComparación de scores:")
print(data[["score", "score_alt"]])

# ¿Cambió el ranking?
# ¿Qué implica esto para el negocio?