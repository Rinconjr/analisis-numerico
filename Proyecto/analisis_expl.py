import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
dataset = pd.read_csv('dataset_desercion_estudiantil.csv', delimiter=';')

# 1. Estadísticas Descriptivas
print("Estadísticas descriptivas del dataset:")
print(dataset.describe())

# 2. Revisión de datos nulos o faltantes
print("\nRevisión de datos faltantes:")
print(dataset.isnull().sum())

# 3. Distribución de variables numéricas
plt.figure(figsize=(10, 6))
dataset.hist(bins=10, figsize=(10, 8), grid=False, edgecolor='black')
plt.tight_layout()
plt.show()

# 4. Distribución de variables categóricas (Situacion_economica y Situacion_familiar)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Situacion_economica', data=dataset)
plt.title('Distribución de Situacion Economica')

plt.subplot(1, 2, 2)
sns.countplot(x='Situacion_familiar', data=dataset)
plt.title('Distribución de Situacion Familiar')
plt.tight_layout()
plt.show()

# 5. Boxplots para detectar outliers en las variables numéricas
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
sns.boxplot(y=dataset['Materias_inscritas'])
plt.title('Materias inscritas')

plt.subplot(1, 3, 2)
sns.boxplot(y=dataset['Promedio_estudiantil'])
plt.title('Promedio estudiantil')

plt.subplot(1, 3, 3)
sns.boxplot(y=dataset['Horas_estudio_por_semana'])
plt.title('Horas de estudio por semana')

plt.tight_layout()
plt.show()

# 6. Matriz de correlación para variables numéricas
numeric_columns = dataset.select_dtypes(include='number')
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# 7. Relación entre las variables y la deserción
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='Desercion', y='Materias_inscritas', data=dataset)
plt.title('Materias inscritas vs Deserción')

plt.subplot(2, 2, 2)
sns.boxplot(x='Desercion', y='Promedio_estudiantil', data=dataset)
plt.title('Promedio estudiantil vs Deserción')

plt.subplot(2, 2, 3)
sns.boxplot(x='Desercion', y='Horas_estudio_por_semana', data=dataset)
plt.title('Horas de estudio por semana vs Deserción')

plt.subplot(2, 2, 4)
sns.boxplot(x='Desercion', y='Edad', data=dataset)
plt.title('Edad vs Deserción')

plt.tight_layout()
plt.show()
