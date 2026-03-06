# Proyecto: Preparación de datos para e-commerce

## Resumen del flujo de trabajo

Este documento describe el proceso completo de preparación de datos realizado para una empresa de e-commerce, utilizando Python y las librerías NumPy y Pandas. El objetivo fue transformar datos crudos de múltiples fuentes en un dataset limpio y estructurado, listo para análisis y modelado.

---

### 1. Justificación del uso de NumPy y Pandas

- **NumPy**: Se utilizó para la generación inicial de datos numéricos (edades, montos, etc.) debido a su eficiencia en operaciones vectorizadas. NumPy trabaja con arrays homogéneos en memoria contigua, lo que permite cálculos rápidos (suma, media, etc.) sin necesidad de bucles explícitos en Python. Esto es crucial cuando se manejan grandes volúmenes de datos.

- **Pandas**: Una vez generados los datos, se empleó Pandas para convertirlos en DataFrames, lo que facilitó la exploración, limpieza y transformación. Pandas ofrece métodos integrados para fusión (`merge`), agregación (`groupby`), reestructuración (`pivot`, `melt`) y manejo de valores nulos, lo que acelera el flujo de trabajo y mejora la legibilidad del código.

---

### 2. Descripción del dataset generado y fuentes externas integradas

- **Datos generados con NumPy**:
  - 200 clientes con `cliente_id`, `edad` (10% nulos intencionales), `país` y `años_membresía`.
  - 1000 transacciones con `transaccion_id`, `cliente_id`, `fecha` (año 2023), `producto_id`, `cantidad`, `precio_unitario` y `monto_total` calculado.

- **Fuentes externas**:
  - **Archivo Excel**: Se creó `productos_ecommerce.xlsx` con 50 productos, incluyendo `nombre_producto`, `categoría` y `proveedor`.
  - **Tabla web**: Se extrajo de Wikipedia una tabla con el PIB nominal por país, usando `pd.read_html()`. Los nombres de país se mapearon para coincidir con los del dataset principal.

La unificación se realizó mediante sucesivos `merge`:
  - Transacciones + clientes (por `cliente_id`).
  - Resultado + productos (por `producto_id`).
  - Resultado + PIB (por `país`).

---

### 3. Técnicas aplicadas para la limpieza y transformación

#### a) Valores nulos
- **Edad**: imputación con la mediana para no perder registros.
- **Producto**: se eliminaron filas sin `nombre_producto` (eran pocas).
- **PIB**: se dejaron como NaN, ya que podrían ser relevantes en análisis posteriores.

#### b) Outliers
- Se utilizó **Z-score** sobre la columna `monto_total`, eliminando registros con |Z| > 3 (considerados extremos y posiblemente erróneos).

#### c) Data Wrangling
- **Duplicados**: se eliminaron registros duplicados.
- **Tipos de datos**: se convirtió `edad` a entero.
- **Nuevas columnas**: `mes`, `año`, `total_con_iva` (monto +21%), `grupo_edad` (Joven/Adulto/Senior) mediante `apply`, `pais` en mayúsculas con `map`.
- **Discretización**: se creó `categoria_monto` (Bajo/Medio/Alto) usando `pd.cut`.

#### d) Agrupamiento y pivoteo
- `groupby` para obtener ventas totales, promedio y conteo por categoría de producto.
- `pivot_table` para generar una matriz de ventas por producto y mes.
- `melt` para demostrar la reversión de un pivot (ejemplo con subconjunto).
- Nuevo `merge` con una tabla de tipo de cambio ficticio por mes.

---

### 4. Principales decisiones y desafíos encontrados

- **Mapeo de países**: Los nombres en la tabla web no coincidían exactamente con los del dataset (ej. "Brazil" vs "Brasil"), por lo que fue necesario un mapeo manual.
- **Selección de tabla web**: La página de Wikipedia contenía múltiples tablas; se identificó la correcta (FMI) por inspección.
- **Outliers**: Se optó por eliminar los outliers extremos (Z > 3) en lugar de transformarlos, asumiendo que representaban errores de carga o transacciones no representativas.
- **Valores nulos en PIB**: Se decidió no imputarlos para no introducir sesgos; podrían ser tratados en fases posteriores según el análisis.

---

### 5. Resultados obtenidos y estado final del dataset

- **Dimensiones finales**: Aproximadamente 1000 filas y 33 columnas (dependiendo de los outliers eliminados).
- **Calidad**: Datos sin duplicados, tipos correctos, valores nulos controlados y outliers tratados.
- **Formatos exportados**: `dataset_final.csv` y `dataset_final.xlsx`, listos para ser utilizados en reportes o modelos de machine learning.

El dataset resultante cumple con los requerimientos de calidad y estructura, y está preparado para análisis exploratorio, visualizaciones o alimentar algoritmos predictivos.