"""
Proyecto: Preparación de datos con Python - E-commerce
Evaluación del módulo 4

Este script integra las 6 lecciones del proyecto:
1. Generación de datos con NumPy
2. Exploración con Pandas
3. Integración de fuentes (CSV, Excel, web)
4. Limpieza (nulos y outliers)
5. Data Wrangling
6. Agrupamiento, pivoteo y exportación final

Autor: Estudiante
Fecha: 2025
"""

# =============================================================================
# Importaciones y configuración inicial
# =============================================================================
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Fijar semillas para reproducibilidad
np.random.seed(42)
random.seed(42)

# =============================================================================
# Lección 1 - La librería NumPy
# Objetivo: Crear datos ficticios de clientes y transacciones con NumPy.
# =============================================================================
print("=" * 60)
print("LECCIÓN 1: Generación de datos con NumPy")
print("=" * 60)

# Generar datos de clientes
n_clientes = 200
clientes_id = np.arange(1, n_clientes + 1)
edades = np.random.randint(18, 70, size=n_clientes).astype(float)
# Introducir algunos valores nulos en edad (simulando datos faltantes)
edades[np.random.choice(n_clientes, 10, replace=False)] = np.nan

paises = np.random.choice(['Argentina', 'Brasil', 'Chile', 'Colombia', 'México'], size=n_clientes)
anios_membresia = np.random.randint(0, 15, size=n_clientes)

# Generar datos de transacciones
n_transacciones = 1000
transacciones_id = np.arange(1, n_transacciones + 1)
clientes_id_trans = np.random.choice(clientes_id, size=n_transacciones)

# Fechas aleatorias en 2023
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
dias = (end_date - start_date).days
fechas = [start_date + timedelta(days=np.random.randint(0, dias)) for _ in range(n_transacciones)]
fechas = np.array(fechas, dtype='datetime64')

productos_id = np.random.randint(1, 51, size=n_transacciones)  # 50 productos
cantidades = np.random.randint(1, 10, size=n_transacciones)
precios_unitarios = np.round(np.random.uniform(10, 500, size=n_transacciones), 2)

# Calcular monto total (operación vectorizada)
montos_totales = cantidades * precios_unitarios

# Estadísticas básicas con NumPy
print("Estadísticas de montos de transacciones (NumPy):")
print(f"  Suma total: {np.sum(montos_totales):.2f}")
print(f"  Media: {np.mean(montos_totales):.2f}")
print(f"  Mediana: {np.median(montos_totales):.2f}")
print(f"  Desviación estándar: {np.std(montos_totales):.2f}")

# =============================================================================
# Lección 2 - La librería Pandas
# Objetivo: Convertir a DataFrames, explorar y guardar CSV preliminar.
# =============================================================================
print("\n" + "=" * 60)
print("LECCIÓN 2: Exploración con Pandas")
print("=" * 60)

# Crear DataFrames
df_clientes = pd.DataFrame({
    'cliente_id': clientes_id,
    'edad': edades,
    'pais': paises,
    'anios_membresia': anios_membresia
})

df_transacciones = pd.DataFrame({
    'transaccion_id': transacciones_id,
    'cliente_id': clientes_id_trans,
    'fecha': fechas,
    'producto_id': productos_id,
    'cantidad': cantidades,
    'precio_unitario': precios_unitarios,
    'monto_total': montos_totales
})

# Unir ambos DataFrames (enriquecer transacciones con datos de cliente)
df_combinado = pd.merge(df_transacciones, df_clientes, on='cliente_id', how='left')

# Exploración inicial
print("Primeras 5 filas:")
print(df_combinado.head())
print("\nÚltimas 5 filas:")
print(df_combinado.tail())
print("\nInformación general:")
df_combinado.info()
print("\nEstadísticas descriptivas:")
print(df_combinado.describe(include='all'))

# Filtro condicional: transacciones con monto > 2000
filtro = df_combinado[df_combinado['monto_total'] > 2000]
print(f"\nTransacciones con monto > 2000: {len(filtro)} registros")

# Guardar DataFrame preliminar en CSV
df_combinado.to_csv('datos_generados.csv', index=False)
print("\nArchivo 'datos_generados.csv' guardado.")

# =============================================================================
# Lección 3 - Obtención de datos desde archivos
# Objetivo: Integrar CSV, Excel y tabla web en un único DataFrame.
# =============================================================================
print("\n" + "=" * 60)
print("LECCIÓN 3: Integración de múltiples fuentes")
print("=" * 60)

# 1. Cargar el CSV generado
df_principal = pd.read_csv('datos_generados.csv', parse_dates=['fecha'])

# 2. Crear un archivo Excel con información complementaria de productos
productos_info = pd.DataFrame({
    'producto_id': range(1, 51),
    'nombre_producto': [f'Producto {i}' for i in range(1, 51)],
    'categoria': np.random.choice(['Electrónica', 'Ropa', 'Hogar', 'Deportes', 'Libros'], 50),
    'proveedor': np.random.choice(['Proveedor A', 'Proveedor B', 'Proveedor C'], 50)
})
productos_info.to_excel('productos_ecommerce.xlsx', index=False)
print("Archivo 'productos_ecommerce.xlsx' creado.")

# Leer el Excel
df_productos = pd.read_excel('productos_ecommerce.xlsx')

# 3. Extraer datos de una tabla web (lista de países por PIB nominal desde Wikipedia)
url = 'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)'
try:
    tablas = pd.read_html(url)
    # La tabla relevante suele ser la segunda (FMI)
    df_pib = tablas[1][['Country', 'GDP(US$MM)']].copy()
    df_pib.columns = ['pais', 'pib_millones_usd']
    # Mapear nombres para que coincidan con nuestros datos
    mapeo = {
        'Argentina': 'Argentina',
        'Brazil': 'Brasil',
        'Chile': 'Chile',
        'Colombia': 'Colombia',
        'Mexico': 'México'
    }
    df_pib['pais'] = df_pib['pais'].replace(mapeo)
    df_pib = df_pib[df_pib['pais'].isin(['Argentina', 'Brasil', 'Chile', 'Colombia', 'México'])]
    print("Datos de PIB extraídos correctamente.")
except Exception as e:
    print(f"Error al leer la tabla web: {e}")
    # Crear un DataFrame vacío como fallback
    df_pib = pd.DataFrame(columns=['pais', 'pib_millones_usd'])

# 4. Unificar fuentes
df_unificado = pd.merge(df_principal, df_productos, on='producto_id', how='left')
df_unificado = pd.merge(df_unificado, df_pib, on='pais', how='left')

# Guardar DataFrame consolidado
df_unificado.to_csv('datos_consolidados.csv', index=False)
print("Archivo 'datos_consolidados.csv' guardado con datos unificados.")

# =============================================================================
# Lección 4 - Manejo de valores perdidos y outliers
# Objetivo: Limpiar nulos y detectar outliers.
# =============================================================================
print("\n" + "=" * 60)
print("LECCIÓN 4: Limpieza de datos (nulos y outliers)")
print("=" * 60)

df = pd.read_csv('datos_consolidados.csv', parse_dates=['fecha'])

# 1. Identificar valores nulos
print("Valores nulos por columna:")
print(df.isnull().sum())

# 2. Gestión de nulos
# - Edad: imputar con la mediana
mediana_edad = df['edad'].median()
df['edad'].fillna(mediana_edad, inplace=True)
# - Eliminar filas sin nombre_producto (si las hubiera)
df.dropna(subset=['nombre_producto'], inplace=True)

# 3. Detección de outliers en 'monto_total' usando Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df['monto_total'].dropna()))
outliers_z = df.iloc[(z_scores > 3).nonzero()[0]] if len(z_scores) > 0 else pd.DataFrame()
print(f"\nOutliers detectados con Z-score (>3): {len(outliers_z)} registros")

# Decisión: eliminar outliers extremos
df_clean = df.drop(outliers_z.index) if not outliers_z.empty else df
print(f"Tamaño del dataset después de eliminar outliers: {df_clean.shape}")

# Guardar dataset limpio
df_clean.to_csv('datos_limpios.csv', index=False)
print("Archivo 'datos_limpios.csv' guardado.")

# =============================================================================
# Lección 5 - Data Wrangling
# Objetivo: Transformar y enriquecer los datos.
# =============================================================================
print("\n" + "=" * 60)
print("LECCIÓN 5: Data Wrangling")
print("=" * 60)

df = pd.read_csv('datos_limpios.csv', parse_dates=['fecha'])

# 1. Eliminar duplicados
dup_antes = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"Duplicados eliminados: {dup_antes}")

# 2. Transformar tipos de datos
df['edad'] = df['edad'].astype(int)

# 3. Crear nuevas columnas calculadas
df['mes'] = df['fecha'].dt.month
df['año'] = df['fecha'].dt.year
df['total_con_iva'] = df['monto_total'] * 1.21  # 21% de IVA

# 4. Aplicar función personalizada para categorizar edad
def categoria_edad(edad):
    if edad < 30:
        return 'Joven'
    elif edad < 50:
        return 'Adulto'
    else:
        return 'Senior'

df['grupo_edad'] = df['edad'].apply(categoria_edad)

# 5. Usar map para estandarizar país (mayúsculas)
df['pais'] = df['pais'].map(lambda x: x.upper())

# 6. Discretizar 'monto_total' en categorías
bins = [0, 500, 1500, df['monto_total'].max()]
labels = ['Bajo', 'Medio', 'Alto']
df['categoria_monto'] = pd.cut(df['monto_total'], bins=bins, labels=labels)

# Guardar versión wrangled
df.to_csv('datos_wrangled.csv', index=False)
print("Archivo 'datos_wrangled.csv' guardado.")
print("\nPrimeras filas del dataset wrangled:")
print(df.head())

# =============================================================================
# Lección 6 - Agrupamiento y pivoteo de datos
# Objetivo: Organizar datos mediante groupby, pivot, melt y exportar final.
# =============================================================================
print("\n" + "=" * 60)
print("LECCIÓN 6: Agrupamiento, pivoteo y exportación final")
print("=" * 60)

df = pd.read_csv('datos_wrangled.csv', parse_dates=['fecha'])

# 1. Agrupamiento: ventas por categoría de producto
ventas_por_categoria = df.groupby('categoria')['monto_total'].agg(['sum', 'mean', 'count']).reset_index()
print("\nVentas por categoría:")
print(ventas_por_categoria)

# 2. Pivot: ventas totales por producto y mes
pivot_ventas = df.pivot_table(values='monto_total', index='producto_id', columns='mes', aggfunc='sum', fill_value=0)
print("\nPivot de ventas por producto y mes (primeras 5 filas):")
print(pivot_ventas.head())

# 3. Melt: ejemplo con un subconjunto del pivot
pivot_sample = pivot_ventas.iloc[:5, :3].reset_index()
melted = pd.melt(pivot_sample, id_vars=['producto_id'], var_name='mes', value_name='ventas')
print("\nEjemplo de melt (revertir pivot):")
print(melted)

# 4. Combinar nueva fuente: tipo de cambio ficticio por mes
tipo_cambio = pd.DataFrame({
    'mes': range(1, 13),
    'usd_a_eur': np.random.uniform(0.85, 0.95, 12)
})
df_final = pd.merge(df, tipo_cambio, on='mes', how='left')

# 5. Exportar dataset final en CSV y Excel
df_final.to_csv('dataset_final.csv', index=False)
df_final.to_excel('dataset_final.xlsx', index=False)
print("\nArchivos finales 'dataset_final.csv' y 'dataset_final.xlsx' exportados.")

print("\n¡Proyecto completado con éxito!")