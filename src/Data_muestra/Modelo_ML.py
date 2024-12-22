#!/usr/bin/env python
# coding: utf-8

# ## Modelo de busqueda de anomalias para documento tipo 33

# In[1]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# In[2]:


##Se importan packages necesarios
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
import pyspark
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from pyspark.sql.types import StringType,TimestampType
import matplotlib.pyplot as plt
from pyspark.sql import functions as F


# In[3]:


#inicio de sesion en spark
ss_name = 'Lectura de datos Dashboard'
wg_conn = "spark.kerberos.access.hadoopFileSystems"
db_conn = "abfs://data@datalakesii.dfs.core.windows.net/"

spark = SparkSession.builder \
      .appName(f"Ejecucion algoritmo {ss_name}")  \
      .config(wg_conn, db_conn) \
      .config("spark.executor.memory", "6g") \
      .config("spark.driver.memory", "12g")\
      .config("spark.executor.cores", "4") \
      .config("spark.executor.instances", "5") \
      .config("spark.driver.maxResultSize", "12g") \
      .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

spark.conf.set("spark.sql.parquet.enableVectorizedReader","false")
spark.conf.set("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
spark.conf.set("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")


# In[4]:


ruta = "abfs://data@datalakesii.dfs.core.windows.net/DatosOrigen/lr-629/APA/Analisis_factura/dataset_final_estratificado"

# Leer el DataFrame desde la ruta especificada
dte = spark.read.format("parquet").load(ruta)


# In[5]:


# Vemos el numeor de filas de dte
dte.count()


# In[6]:


dte.columns


# In[7]:


dte=dte.toPandas()


# In[8]:


spark.stop()


# In[9]:


df=dte


# In[10]:


#Se definen todas las variables del modelo, categoricas y  numericas

numerical_columns = [
    'dhdr_mnt_total', 'dhdr_iva',
    'recep_COCO_IMP_VENTAS_IVA',
    'recep_COCO_IMP_VENTAS_TRANSPORTE',
    'recep_COCO_MTO_DEV_SOLICITADA_F22',
    'recep_COCO_MTO_VENTAS',
    'recep_NEGO_NRO_FACTURAS_6MESES_VO',
    'emis_COCO_IMP_VENTAS_IVA',
    'emis_COCO_IMP_VENTAS_TRANSPORTE',
    'emis_COCO_MTO_DEV_SOLICITADA_F22',
    'emis_COCO_MTO_VENTAS',
    'emis_NEGO_NRO_FACTURAS_6MESES_VO',
    'avg_dhdr_mnt_total_emisor',
     'stddev_dhdr_mnt_total_emisor',
     'avg_dhdr_iva_emisor',
     'stddev_dhdr_iva_emisor',
     'avg_dhdr_mnt_total_receptor',
     'stddev_dhdr_mnt_total_receptor',
     'avg_dhdr_iva_receptor',
     'stddev_dhdr_iva_receptor'
]



categorical_columns = [
    'es_fin_de_semana',
     'bloque_horario', 'dia_semana', 'semana_mes',
    'emis_INICIO_SEGMENTO', 'emis_ACEC_DES_SUBRUBRO_PPAL', 'emis_Alerta_1019', 'emis_Alerta_2250',
    'emis_Alerta_400X', 'emis_Alerta_4110', 'emis_Alerta_4111', 'emis_Alerta_4112',
    'emis_Alerta_4113', 'emis_Alerta_52', 
#    'emis_Alerta_5201',
    'emis_Alerta_5203',
    'emis_Alerta_53',
#    'emis_Alerta_5301',
    'recep_INICIO_SEGMENTO', 'recep_ACEC_DES_SUBRUBRO_PPAL', 'recep_Alerta_1019', 'recep_Alerta_2250',
    'recep_Alerta_400X', 'recep_Alerta_4110', 'recep_Alerta_4111', 'recep_Alerta_4112',
    'recep_Alerta_4113', 'recep_Alerta_52',
#    'recep_Alerta_5201',
    'recep_Alerta_5203',
 #   'recep_Alerta_5301',
    'recep_Alerta_53'
]



# columnas de alerta. 
alert_columns = [
    'emis_Alerta_1019', 'emis_Alerta_2250', 'emis_Alerta_400X', 'emis_Alerta_4110', 'emis_Alerta_4111', 
    'emis_Alerta_4112', 'emis_Alerta_4113', 'emis_Alerta_52', 'emis_Alerta_5203', 'emis_Alerta_53', 
    'recep_Alerta_1019', 'recep_Alerta_2250', 'recep_Alerta_400X', 'recep_Alerta_4110', 'recep_Alerta_4111', 
    'recep_Alerta_4112', 'recep_Alerta_4113', 'recep_Alerta_52', 'recep_Alerta_5203', 'recep_Alerta_53'
]


# ## Imputacion de alertas

# In[11]:


# Verificar valores nulos antes de rellenar
null_before = df[alert_columns].isnull().sum()

# Rellenar los valores nulos con 0 para las columnas de alerta
df[alert_columns] = df[alert_columns].fillna(0)

# Verificar valores nulos después de rellenar
null_after = df[alert_columns].isnull().sum()

# Crear un DataFrame para mostrar los valores nulos antes y después
null_comparison = pd.DataFrame({
    'Column': alert_columns,
    'Nulos antes': null_before,
    'Nulos después': null_after
})

# Mostrar la tabla de comparación
print(null_comparison)


# ## Valores nulos en variables categoricas

# In[12]:


# Recuento de valores nulos en columnas categóricas
missing_values_categorical = df[categorical_columns].isnull().sum()
print("\nRecuento de valores nulos en columnas categóricas:")
print(missing_values_categorical)


# In[13]:


# Recuento de valores nulos en columnas numéricas
missing_values_numerical = df[numerical_columns].isnull().sum()
print("\nRecuento de valores nulos en columnas numéricas:")
print(missing_values_numerical)


# In[14]:


# Imputar valores nulos con cero en las columnas numéricas
df[numerical_columns] = df[numerical_columns].fillna(0)


# In[15]:


# Se reemplazan los valores faltantes (NaN) en las columnas relevantes con el valor 'Desconocido',
# para asegurar que no haya valores nulos en el DataFrame y que se pueda trabajar con datos completos.
df['recep_INICIO_SEGMENTO'] = df['recep_INICIO_SEGMENTO'].fillna('Desconocido')
df['emis_INICIO_SEGMENTO'] = df['emis_INICIO_SEGMENTO'].fillna('Desconocido')

df['emis_ACEC_DES_SUBRUBRO_PPAL'] = df['emis_ACEC_DES_SUBRUBRO_PPAL'].fillna('Desconocido')
df['recep_ACEC_DES_SUBRUBRO_PPAL'] = df['recep_ACEC_DES_SUBRUBRO_PPAL'].fillna('Desconocido')


# In[16]:


columns_to_keep = numerical_columns + categorical_columns + ['dhdr_folio']

# Actualizar el DataFrame solo con esas columnas
df = df[columns_to_keep]


# In[17]:


#df=df.drop(columns=['emis_Alerta_5301', 'emis_Alerta_5201','recep_Alerta_5301', 'recep_Alerta_5201'])


# In[18]:


print(df['recep_INICIO_SEGMENTO'].unique())


# In[19]:


# Obtener todas las columnas del DataFrame
all_columns = numerical_columns + categorical_columns


# In[20]:


# Filtrar las columnas categóricas (excluyendo 'dhdr_folio' y las numéricas)
categorical_cols = [col for col in all_columns if col not in numerical_columns and col != 'dhdr_folio']

# Mostrar las columnas categóricas
print("Columnas categóricas:", categorical_cols)

# Realizar one-hot encoding en las columnas categóricas
df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)

# Crear el objeto StandardScaler para estandarizar las variables numéricas
scaler = StandardScaler()

# Estandarizar las columnas numéricas (sin modificar el DataFrame original)
df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)

# Incluir 'dhdr_folio' como identificador en el dataset final (sin transformarlo)
df_final = pd.concat([df[['dhdr_folio']], df_scaled, df_encoded], axis=1)


# In[21]:


# Aplicar Isolation Forest para detectar outliers
iso_forest = IsolationForest(contamination=0.1, random_state=42)  # Ajusta el parámetro de contaminación según tu caso
outliers = iso_forest.fit_predict(df_final.drop('dhdr_folio', axis=1))  # No incluir 'dhdr_folio' en el fit

# Los outliers estarán marcados como -1 (outliers) y 1 (no outliers)
df_final['outliers'] = outliers

# Contar cuántos outliers hay
outliers_count = (df_final['outliers'] == -1).sum()

# Visualizar los primeros resultados
print(df_final[['dhdr_folio', 'outliers']].head())
print(f"Total de outliers detectados: {outliers_count}")


# In[22]:


df_final.shape


# ## Analisis de anomalias del dataframe original

# In[23]:


# Agregar la columna 'outliers' de df_final al DataFrame original df usando .loc
df.loc[:, 'outliers'] = df_final['outliers']

# Ver las primeras filas del DataFrame original con la columna de outliers
print(df[['dhdr_folio', 'outliers']].head())


# In[24]:


df.columns


# In[28]:


spark.stop()


# In[ ]:





# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text  # Asegúrate de importar esto
from sklearn.metrics import classification_report, accuracy_score

# Dividir el DataFrame en variables numéricas y categóricas
X_numerical = df[numerical_columns]  # Variables numéricas
X_categorical = df[categorical_columns]  # Variables categóricas

# Realizar One-Hot Encoding de las variables categóricas
X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True)  # Evitar la trampa de las variables ficticias

# Concatenar las variables numéricas y las categóricas codificadas
X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1)

# Asegurarse de que todas las variables numéricas sean del tipo correcto (float32)
X_processed = X_processed.astype(np.float32)

# Definir la variable dependiente (outliers)
y = df['outliers']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Crear el modelo de Decision Tree
dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  # Limitar la profundidad del árbol para evitar sobreajuste
    min_samples_split=10,  # Mínimo de muestras necesarias para dividir un nodo
    min_samples_leaf=5,  # Mínimo de muestras necesarias en una hoja
    class_weight='balanced'  # Ajustar los pesos de las clases para manejar desbalance
)

# Entrenar el modelo
dt.fit(X_train, y_train)

# Predicción y evaluación
y_pred = dt.predict(X_test)

# Evaluar el modelo
print("Accuracy del modelo:", accuracy_score(y_test, y_pred))
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))


# Para la clase -1 (Outliers):
# 
# Precision: 0.42
# Esto indica que el 42% de las instancias predichas como outliers son realmente outliers. Este valor es relativamente bajo, lo que sugiere que el modelo está generando bastantes falsos positivos (instancias normales clasificadas como outliers).
# 
# Recall: 0.77
# El 77% de los verdaderos outliers son correctamente identificados. Este es un buen valor, ya que significa que el modelo no está perdiendo demasiados outliers, aunque todavía podría mejorar.
# 
# F1-Score: 0.54
# El F1-Score es el promedio armónico entre precision y recall. Un F1 de 0.54 para los outliers es moderado, lo que indica que el modelo tiene un rendimiento intermedio al balancear ambos aspectos.

# In[ ]:


# Mostrar las reglas del árbol
print("Reglas del árbol de decisión:")
tree_rules = export_text(dt, feature_names=list(X_processed.columns))
print(tree_rules)

# Acceder a las propiedades del árbol
tree = dt.tree_

# Obtener los índices de las hojas (convertir X_test a un numpy.ndarray)
leaf_indices = tree.apply(X_test.values)  # Convertir X_test a ndarray

# Contar el número de ejemplos de cada clase en cada hoja
leaf_classes = np.zeros((tree.node_count,), dtype=int)  # Array para almacenar la clase predominante en cada hoja

# Contar las instancias por hoja
for i in range(len(leaf_indices)):
    leaf_classes[leaf_indices[i]] += 1

# Buscar la hoja con más clasificación correcta
leaf_purity = {}
for leaf in np.unique(leaf_indices):
    # Obtener el índice de la hoja
    leaf_purity[leaf] = np.sum(y_test[leaf_indices == leaf] == y_pred[leaf_indices == leaf])

# Imprimir la hoja con mayor pureza
best_leaf = sorted(leaf_purity.items(), key=lambda x: x[1], reverse=True)[0][0]

print(f"La hoja con mejor clasificación tiene el índice {best_leaf} con una pureza de {leaf_purity[best_leaf]} ejemplos correctos.")

# Mostrar más detalles sobre la hoja seleccionada
print(f"Detalles de la hoja de mejor clasificación: {best_leaf}")
print(f"Predicción de la hoja: {dt.tree_.value[best_leaf]}")


# 
# 
# El modelo ha mostrado la predicción de la hoja como [[0.09855173, 0.90144827]]. Esto es un vector de probabilidades, donde cada valor representa la probabilidad de que una instancia en esa hoja pertenezca a cada clase (en este caso, parece que es un problema binario).
# 
# 0.09855173 es la probabilidad de que la instancia pertenezca a la clase 0 (por ejemplo, "No es un outlier").
# 
# 0.90144827 es la probabilidad de que la instancia pertenezca a la clase 1 (por ejemplo, "Es un outlier").
# 
# Dado que la suma de las probabilidades es 1, la probabilidad de la clase 1 es mucho mayor que la de la clase 0. Esto indica que, en promedio, las instancias en esta hoja están más inclinadas a ser clasificadas como "outliers" (si esa es la clase 1).

# In[30]:


#REGLAS COMPLETAS DEL ARBOL DE DECISION

from sklearn.tree import _tree

def get_tree_rules(tree, feature_names):
    """
    Función para obtener las reglas de decisión de un árbol de decisión.
    Retorna un diccionario con las reglas para cada nodo.
    """
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" 
                    for i in tree_.feature]
    
    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]
            
            # Regla para el nodo
            rule = f"Si {name} <= {threshold:.2f}, ir a la izquierda, de lo contrario, ir a la derecha."
            
            # Recursión para los nodos hijo
            left_rule = recurse(left_child)
            right_rule = recurse(right_child)
            
            return f"{rule} \n[Izquierda]: {left_rule} \n[Derecha]: {right_rule}"
        else:
            # Nodo hoja: retornar la clase mayoritaria
            return f"Clase {tree_.value[node].argmax()} (Valor: {tree_.value[node]})"

    # Iniciar recursión desde la raíz
    return recurse(0)

# Obtener las reglas del árbol
rules = get_tree_rules(dt, list(X_processed.columns))

# Mostrar las reglas completas
print("Reglas completas del árbol de decisión:\n", rules)


# In[ ]:


from sklearn.tree import _tree

def get_decision_path(tree, feature_names, leaf_index):
    """
    Devuelve el camino desde la raíz hasta la hoja especificada.
    """
    tree_ = tree.tree_

    # Convertir los índices de las características en nombres
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" 
                    for i in tree_.feature]

    # Inicializamos el camino y el nodo actual (la hoja que nos interesa)
    path = []
    node = leaf_index

    # Seguir el camino hacia atrás desde la hoja hasta la raíz
    while node != 0:  # Continuamos hasta llegar a la raíz
        # Obtenemos las condiciones de los nodos
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Si la condición es a la izquierda o a la derecha
            if tree_.children_left[node] == node:
                path.append(f"Si {name} <= {threshold:.2f}")
            else:
                path.append(f"Si {name} > {threshold:.2f}")
        
        # Retrocedemos al nodo padre
        node = tree_.children_left[node] if tree_.children_left[node] != _tree.TREE_UNDEFINED else tree_.children_right[node]
    
    # Regresamos el camino en el orden correcto (de arriba a abajo)
    return path[::-1]

# Obtener el índice de la hoja de decisión (en este caso, 20)
leaf_index = 20

# Obtener las reglas que conducen a la hoja 20
leaf_rules = get_decision_path(dt, list(X_processed.columns), leaf_index)

# Mostrar las reglas
print("Reglas para la hoja con el índice 20:")
print(" -> ".join(leaf_rules))



# In[ ]:




