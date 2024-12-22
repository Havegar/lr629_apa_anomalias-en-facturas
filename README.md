## Tabla de contenidos
1. [General Info](#general-info)
2. [Tecnologias](#Tecnologias)
3. [Recursos CML](#Recursos_CML)
4. [Colaboradores](#Colaboradores)
5. [Instalación](#Instalación)
6. [Ejecución](#Ejecución)
7. [Explicación](#Explicación)
8. [Sugerencias y pasos siguientes](#Sugerencias)
9. [Estructura de carpetas](#plantilla_proyectos_python)
10. [Estructura de tablas y rutas](#Estructura_rutas_tablas)


# Detección de anomalías en facturas

El proyecto de Análisis de Documentos Tipo 33 tiene como objetivo identificar patrones anómalos en estos documentos mediante el análisis de características específicas de los emisores y del propio documento. Se examinan factores como el historial de emisión, la frecuencia, la ubicación geográfica y el sector económico del emisor, además de detalles temporales y geográficos asociados al documento.

El proyecto utiliza tanto métodos descriptivos como técnicas de machine learning no supervisado para detectar irregularidades sin necesidad de etiquetas previas, lo que permite identificar comportamientos atípicos y agrupaciones de documentos que podrían considerarse inusuales. De este modo, se generan indicadores que ayudan a detectar documentos Tipo 33 anómalos, aportando información valiosa para un análisis y control más riguroso en este ámbito.


##  Fases del proyecto

El proyecto de Análisis de Documentos Tipo 33 se desarrolla en dos fases principales. La primera fase se centra en la recolección de datos, cuyo objetivo es reunir todas las fuentes relevantes para el análisis. Esto incluye no solo los documentos Tipo 33, sino también una variedad de datos adicionales sobre las características de los contribuyentes, que deben ser anexadas a cada documento para obtener un contexto más completo. Este proceso permite construir una base de datos robusta y detallada, que también incorpora otros documentos vinculados al análisis.

La segunda fase aborda el análisis propiamente dicho, que incluye tanto el estudio de los documentos Tipo 33 como la búsqueda de patrones y la identificación de posibles anomalías en los datos. En esta etapa se emplean técnicas de machine learning y métodos descriptivos, permitiendo detectar comportamientos inusuales y patrones atípicos en los documentos y en los datos agregados de contribuyentes.

## Tecnologías Utilizadas

### Apache Spark
- *Versión*: 3.2.3
- *Uso*: Consulta de tablas ubicadas en el Data Lake.
- *Configuración*:
  - Memoria de ejecución: 16 GB
  - Memoria del driver: 12 GB
  - Núcleos de ejecución: 2
  - Instancias de ejecución: 2

### Python
- *Versión*: 3.10
- *Uso*: Procesamiento adicional en Cloudera Machine Learning (CML).
- *Tipo de sesi*: Por otro lado y acerca de la ejecución del proyecto, se requiere en particular que las sesiones de trabajo tengan una memoria ram de 96 gb y 7 CPU como mínimo.


### Packages de Python mas utilizados
- *pandas*: Manejo de dataframes
- *numpy*: Operaciones matemáticas
- *pyspark*: Para interactuar con Spark desde Python.
- *seaborn*: Visualización de datos.
- *matplotlib*: Generación de gráficos y visualizaciones.
- *warnings*: Gestión de mensajes de advertencia.
- *json*: Manipulación de datos en formato JSON.
- *os*: Interacción con el sistema operativo.
- *sklearn*: librería de Python para aprendizaje automático, que ofrece algoritmos eficientes para clasificación, regresión, agrupamiento y más, facilitando el análisis y modelado de datos.
- *pyarrow*:  librería de Python para trabajar con el formato de datos en memoria Apache Arrow, optimizando el procesamiento y la interoperabilidad entre diferentes sistemas y lenguajes.



El proyecto de Análisis de Documentos Tipo 33 se apoya en los recursos de Cloudera Machine Learning (CML), utilizando herramientas y procesos que permiten un análisis eficiente de los datos. Todo el trabajo se ejecuta a través de notebooks de Python, que se procesan de manera secuencial, siguiendo un flujo específico que se detallará más adelante. Estos notebooks están conectados al Data Lake, lo que facilita el acceso a los datos y asegura un almacenamiento optimizado para grandes volúmenes de información.

En la primera fase, los notebooks de Python se utilizan para obtener y preparar los datos. Se extraen tanto los documentos Tipo 33 como las características de los contribuyentes que deben ser anexadas a cada documento. Se realiza un muestreo estratificado basado en el tipo de contribuyente emisor, así como un muestreo de la data completa en una ventana de tiempo específica. Los resultados se almacenan en el Data Lake, asegurando una infraestructura segura y escalable.

En la segunda fase, el análisis se realiza mediante los notebooks de Python, donde se emplean tanto técnicas de machine learning no supervisado como análisis exploratorio de datos. El dataset estratificado se utiliza para entrenar modelos de machine learning que ayudan a detectar patrones anómalos en los documentos, mientras que el dataset completo se usa para realizar un análisis exploratorio más profundo, identificando características y posibles irregularidades. Todo este proceso se lleva a cabo dentro del entorno de Cloudera Machine Learning, lo que optimiza el rendimiento y la gestión de los recursos del sistema.


### Generación del Dashboard:
En un notebook adicional, se utiliza Gradio para generar el dashboard que visualiza los datos recopilados y procesados.
El dashboard permite a los usuarios interactuar con los datos de manera dinámica y tomar decisiones informadas basadas en los indicadores presentados.

## Colaboradores
***
En este proyecto participa Henry Vega, Data Analyst de APIUX así como el equipo del Area de además Arnol Garcia, jefe de área y Manuel Barrientos, con soporte técnico en el espacio de trabajo y Jorge Estéfane, jefe de proyecto por parte de APIUX.

## Instalación
***

Los pasos para instalar el proyecto y sus configuraciones en un entorno nuevo en caso que sea necesario, por ejemplo:
1. Instalar paquetes
  - Ingresar a la carpeta package_requeriment y abrir la terminal 
  - Escribir el siguiente comando: pip install -r requirements.txt
2. Ejecutar Notebooks (análisis de grupos económicos y/o búsqueda de grupos económicos) y scripts en sus jobs respectivos.

3. Uso posterior de la data output obtenida. 

## Ejecución del proyecto
************************************************

### Ejecución del Proyecto

El proyecto de análisis de documentos tipo 33 se estructura en diversos procesos que se ejecutan en notebooks de Python. Cada proceso tiene un flujo bien definido y los datos generados en cada etapa se almacenan en carpetas específicas dentro de la estructura del proyecto.

#### 1. **Archivo de Data de Contribuyentes**

Adicionalmente, el archivo **`data-contribuyentes.ipynb`** se utiliza para recolectar los datos de los contribuyentes, que luego se integrarán tanto en el proceso de la submuestra como en el proceso de la data completa. Este archivo es fundamental para enlazar la información de los contribuyentes con los documentos y filtrar los respectivos .

#### 2. **Proceso de Submuestra**

En este proceso, se obtienen los documentos dentro de un periodo de tiempo específico, y posteriormente se estratifican según los contribuyentes emisores. El flujo de notebooks en esta etapa es el siguiente:

- **Data_documentos_muestra_estratificada.ipynb**: Este notebook obtiene los documentos en el periodo determinado y realiza la estratificación, asociando los documentos con los contribuyentes emisores para asegurar que la muestra sea representativa de los distintos tipos de emisores (gran empresa, mediana empresa, persona natural, etc.).
- **Union_documento_contribuyente_estratificado.ipynb**: Este notebook se encarga de realizar la unión de los datos de los documentos con la información de los contribuyentes emisores, creando un dataset estratificado que incluye tanto los documentos como los datos relevantes de los emisores.

Para este proceso, los archivos generados se almacenan en la carpeta **`src/muestra/`**, que contiene toda la información y datos relacionados con la muestra estratificada.

Una vez que los datos han sido estratificados, se procede con el análisis de los mismos mediante **Machine Learning**:

- **Modelo_ML.ipynb**: Este notebook ejecuta el modelo de machine learning para detectar patrones anómalos en los documentos. El análisis permite identificar irregularidades dentro del conjunto de datos. Inicialmente usa un isolation forest para detectar outliers, luego esa clasificacion ingresa a un arbol aleatorio. Luego de ello se analiza la hoja con mejor clasificacion.

#### 3. **Proceso de Data Completa**

En esta etapa, se obtiene toda la data disponible, sin realizar estratificación, y se realiza una unión con los contribuyentes correspondientes. El flujo de notebooks en esta etapa es el siguiente:

- **Data_documentos_no_estratificado.ipynb**: Este notebook obtiene todos los documentos del periodo, sin aplicar estratificación. El proceso incluye la extracción de todos los datos para un análisis general más exhaustivo.
- **Union_documento_contribuyente_no_estratificado.ipynb**: Aquí se realiza la unión de los documentos con los contribuyentes, pero sin segmentar por categorías. El resultado es un dataset completo con los documentos junto con la información de los contribuyentes.

Los archivos generados en esta parte se almacenan en la carpeta **`src/data-completa/`**.

Una vez que los datos de los documentos y los contribuyentes se han unido, se realiza un análisis exploratorio de los datos para identificar patrones y reglas potenciales:

- **Analisis exploratorio de datos y posibles reglas.ipynb**: Este notebook realiza un análisis exploratorio de los datos, en el que se identifican patrones, relaciones y posibles reglas que podrían señalar anomalías o irregularidades. Luego de ello, se hace un analisis de los datos sobre precentil 90% de monto del documento, separando por segmento de emisor. Se analizan las caracteristicas de esas poblaciones para establecer una serie de reglas de esas anomalias.


El orden de ejecucion del proyecto es:

* Ejecucion de notebook de data de contribuyentes.
* Ejecucion de notebook para obtener los documentos.
* Ejecucion de notebook para hacer la union data de contribuyentes, documentos y sus caracteristicas.
* Notebooks de analisis exploratorio de datos y modelo de machine learning. 


### Sugerencias

- **Variables Adicionales**: Hay muchas variables adicionales que se pueden obtener relacionadas con las operaciones entre emisor y receptor en un periodo determinado. Estas variables pueden incluir el total de emisión de un emisor, lo que recibe ese emisor, el total que recibe un receptor, y lo que emite ese receptor. Estas variables agregadas pueden ayudar a identificar patrones o anomalías al relacionar los datos entre emisores y receptores.

- **Corroboración de Integridad de Datos**: Es importante corroborar la integridad y validación de las variables asociadas a las características de los contribuyentes. Esto puede implicar revisar y verificar si los datos están completos, si las relaciones entre variables tienen sentido y si hay registros faltantes o incorrectos. Esta verificación garantiza que los análisis posteriores se basen en datos fiables y consistentes.

- **Exploración de la Data**: Dada la gran cantidad de variables disponibles, la forma en que se exploró la data hasta ahora se centró principalmente en un valor obtenido debido a que inicialmente se pensó que la temporalidad de la data podía ser relevante para identificar montos anómalos. No obstante, existen múltiples criterios expertos para definir qué constituye un valor anómalo. Dependiendo del rubro, tipo de contribuyente, o cualquier otra variable específica, los valores anómalos pueden variar. Se recomienda tener en cuenta estas características al definir las anomalías.

- **Actualización de los Datos**: La data utilizada actualmente está disponible solo hasta un cierto punto del año 2022. Para garantizar la validez y relevancia de los análisis, se recomienda mantener la data actualizada. Esto es esencial para reflejar los cambios recientes en las actividades económicas de los contribuyentes y mejorar la precisión de los modelos predictivos o análisis de anomalías.

- **Consideración de Criterios Adicionales en el Análisis de Anomalías**: Para una mejor identificación de patrones anómalos, sería valioso considerar criterios adicionales en el análisis, como las características particulares del rubro de cada contribuyente, su tamaño, actividad económica o cualquier otra dimensión que pueda influir en los datos. Esto podría mejorar la precisión en la detección de valores anómalos o irregularidades dentro del conjunto de datos.

## Plantilla del proyecto
***
Se especifica a continuacion la estructura del proyecto.

Proyecto/
│			
├── notebooks/          				# Jupyter notebooks para exploración de datos, prototipado de modelos, etc.

│   ├── [noteebok1.ipynb]			

│   ├── [notebook2.ipynb]		

│   ├── [notebook3.ipynb]		

│			
├── src/                				# Código fuente Python

│   ├── data/           				# Módulos para cargar, limpiar y procesar datos.

│   ├── models/         				# Definiciones de modelos

│   ├── evaluation/     				# Scripts para evaluar el rendimiento de los modelos.

│   └── utils/          				# Utilidades y funciones auxiliares.

│			

├── data/        				        # Bases de dato de todo tipo que se utilizan o generan en el proyecto.

│   ├── external/     				    # Data externa

│   ├── processed/          			# Data procesada

│   └── raw/                            # Data cruda

│					

├── requirements.txt    				# Archivo de requisitos para reproducir el entorno de Python.

│			

└── readme.md           				# Descripción general del proyecto y su estructura.

** Archivos entre [ ] son ejemplos solamente y no se entregan por ahora.



