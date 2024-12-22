#!/usr/bin/env python
# coding: utf-8

# ## Proceso ETL data contribuyentes

# En este notebook se obtiene caracteristicas unicas de contribuyentes. Esta informacion es guardara en un archivo parquet el cual sera posteriormente utilizado para caracterizar las transaccion de documentos tributarios electronicos. 

# In[1]:


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


# In[2]:


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
spark.conf.set("spark.sql.debug.maxToStringFields", "2000")


# ## Segmento del contribuyente

# In[3]:


# Cargar las tablas
df_contribuyentes = spark.table("DW.DW_TRN_CONTRIBUYENTES_E")
df_atributos_contrib = spark.table("DW.DW_TRN_RIAC_ATRIBUTO_CONTRIB_E")
df_atributo_dim = spark.table("DW.DW_DIM_ATRIBUTO_CONTRIB")


# In[4]:


# Crear la tabla temporal #SGM_I
df_sgm_i = df_contribuyentes.alias("t1") \
    .join(df_atributos_contrib.alias("t3"), 
           df_contribuyentes["CONT_RUT"] == df_atributos_contrib["CONT_RUT"], 
           "left") \
    .join(df_atributo_dim.alias("t4"), 
           (df_atributos_contrib["TATR_CODIGO"] == df_atributo_dim["TATR_CODIGO"]) & 
           (df_atributos_contrib["ATRC_FECHA_TERMINO"].isNull()) & 
           (df_atributos_contrib["ATRC_VIGENTE"] == 1) & 
           (df_atributos_contrib["TATR_CODIGO"].isin(['SGGC', 'SGME', 'SGMI', 'SGPM', 'SGPE']))) \
    .select(
        F.col("t1.CONT_RUT"),
        F.col("t1.CONT_DV"),
        F.col("t3.ATRC_VIGENTE"),
        F.col("t4.TATR_VIGENTE"),
        F.col("t3.TATR_CODIGO"),
        F.when(F.col("t3.TATR_CODIGO") == 'SGGC', 5)
         .when(F.col("t3.TATR_CODIGO") == 'SGME', 4)
         .when(F.col("t3.TATR_CODIGO") == 'SGPM', 3)
         .when(F.col("t3.TATR_CODIGO") == 'SGMI', 2)
         .when(F.col("t3.TATR_CODIGO") == 'SGPE', 1).alias("SGM_NUM"),
        F.col("t3.ATRC_FECHA_INICIO"),
        (F.year(F.col("t3.ATRC_FECHA_INICIO")) * 10000 +
         F.month(F.col("t3.ATRC_FECHA_INICIO")) * 100 +
         F.dayofmonth(F.col("t3.ATRC_FECHA_INICIO"))).alias("DIA_AGNO"),
        F.col("t4.TATR_DESCRIPCION")
    )

# Crear una vista temporal
df_sgm_i.createOrReplaceTempView("SGM_I")


# In[5]:


df_sgm_i.columns


# In[6]:


df_sgm_i.select('CONT_RUT').distinct().count()


# In[7]:


# Crear la tabla temporal #SGM_FM
df_sgm_fm = spark.sql("""
    SELECT 
        COALESCE(AL1.CONT_RUT, AL2.CONT_RUT) AS CONT_RUT, 
        COALESCE(AL1.CONT_DV, AL2.CONT_DV) AS CONT_DV, 
        AL1.FECHA_INI_M, 
        AL2.ATRC_FECHA_INICIO, 
        AL2.TATR_CODIGO, 
        AL2.SGM_NUM, 
        AL2.DIA_AGNO
    FROM  
        (SELECT 
            A1.CONT_RUT, 
            A1.CONT_DV, 
            MAX(A1.ATRC_FECHA_INICIO) AS FECHA_INI_M
        FROM 
            SGM_I A1
        GROUP BY 
            A1.CONT_RUT, A1.CONT_DV) AL1
    JOIN 
        SGM_I AL2 ON AL1.FECHA_INI_M = AL2.ATRC_FECHA_INICIO 
        AND AL1.CONT_RUT = AL2.CONT_RUT
""")

df_sgm_fm.createOrReplaceTempView("SGM_FM")


# In[8]:


# Crear la tabla temporal #SGM_TRN_FINAL con las columnas adicionales
df_sgm_trn_final = spark.sql("""
    SELECT 
        COALESCE(AL1.CONT_RUT, AL2.CONT_RUT) AS CONT_RUT, 
        COALESCE(AL1.CONT_DV, AL2.CONT_DV) AS CONT_DV, 
        AL2.DIA_AGNO as INICIO_SEGMENTO,
        -- Agregar columna de Segmento de Empresa
        CASE 
            WHEN AL2.TATR_CODIGO = 'SGGC' THEN 'Segmento Grandes Empresas/Contribuyentes'
            WHEN AL2.TATR_CODIGO = 'SGME' THEN 'Segmento Medianas Empresas'
            WHEN AL2.TATR_CODIGO = 'SGPM' THEN 'Segmento Pequeñas Empresas'
            WHEN AL2.TATR_CODIGO = 'SGMI' THEN 'Segmento Micro Empresas'
            ELSE NULL
        END AS ES_EMPRESA,
        -- Agregar columna para indicar si es persona o no
        CASE 
            WHEN AL2.TATR_CODIGO = 'SGPE' THEN 1 -- Si es persona
            ELSE NULL -- Si no es persona
        END AS ES_PERSONA
    FROM 
        (SELECT 
            A1.CONT_RUT, 
            A1.CONT_DV, 
            MAX(A1.SGM_NUM) AS SGM_NUM_MAX
        FROM 
            SGM_FM A1
        GROUP BY 
            A1.CONT_RUT, A1.CONT_DV) AL1
    JOIN 
        SGM_FM AL2 ON AL1.SGM_NUM_MAX = AL2.SGM_NUM 
        AND AL1.CONT_RUT = AL2.CONT_RUT
""")
# Mostrar el resultado
#df_sgm_trn_final=df_sgm_trn_final.sample(withReplacement=False, fraction=0.1).limit(3)
#df_sgm_trn_final.show()


# In[9]:


# Analizamos si tenemos valores unicos de CONT_RUT para evitar tener mas de una fila con informacion por contribuyente

df_sgm_trn_final.select('CONT_RUT').count()


# In[10]:


df_sgm_trn_final.select('CONT_RUT').distinct().count()


# ### Tabla DW.DW_TRN_ALERTAS_E 
# -----------------------------------------------------------------------------------------------------------------

# In[11]:


dw_trn_alertas = spark.sql("""
SELECT 
    CONT_RUT,
    CONT_DV,
    CASE 
        WHEN ALER_COD_TIPO_ALERTA_VO BETWEEN 4001 AND 4007 
        THEN 'DELITO/QUERELLA' 
        ELSE ALER_DESC_TIPO_ALERTA_VO 
    END AS ALER_DESC_TIPO_ALERTA_VO,
    UNOP_UNIDAD_ACTIV,
    ALER_FECHA_ACTIV_VO,
    CASE WHEN ALER_COD_TIPO_ALERTA_VO LIKE '400%' THEN '400X'
    ELSE ALER_COD_TIPO_ALERTA_VO END AS ALER_COD_TIPO_ALERTA_VO
FROM 
    DW.DW_TRN_ALERTAS_E 
WHERE 
    ALER_COD_TIPO_ALERTA_VO IN (4110, 4111, 4112, 4113, 52, 5201, 5203, 53, 5301, 2250, 1019, 4001, 4002, 4004, 4005, 4006, 4007) 
    AND ALER_FECHA_DESACTIV_VO IS NULL
""").distinct()
#count=13351059

# Agrupar por ALER_DESC_TIPO_ALERTA_VO y contar, luego ordenar en orden descendente
agrupacion_alertas = dw_trn_alertas.groupBy("ALER_DESC_TIPO_ALERTA_VO").count().orderBy("count", ascending=False)

#Por el momento y dado que hay un solo registro, dejamps fuera el codigo 2046 de domicilio inexistente  
# Mostrar los resultados
agrupacion_alertas.show()


# In[12]:


# Pivotear los datos y agregar prefijo "Alerta_" a las columnas
pivot_alertas = dw_trn_alertas.groupBy("CONT_RUT", "CONT_DV") \
    .pivot("ALER_COD_TIPO_ALERTA_VO") \
    .agg(F.first("ALER_COD_TIPO_ALERTA_VO"))

# Renombrar las columnas con prefijo "Alerta_"
for col_name in pivot_alertas.columns:
    if col_name not in ["CONT_RUT", "CONT_DV"]:  # Evitar cambiar las columnas de identificación
        pivot_alertas = pivot_alertas.withColumnRenamed(col_name, f"Alerta_{col_name}")

# Transformar los valores a 1 o 0
pivot_alertas = pivot_alertas.select(
    "CONT_RUT",
    "CONT_DV",
    *[(F.when(F.col(col).isNotNull(), 1).otherwise(0).alias(col)) for col in pivot_alertas.columns if col not in ["CONT_RUT", "CONT_DV"]]
).distinct()


#count=13351059
#dw_trn_alertas=pivot_alertas.sample(withReplacement=False, fraction=0.1).limit(3)
dw_trn_alertas=pivot_alertas
dw_trn_alertas.limit(5).show()


# In[13]:


dw_trn_alertas.select('CONT_RUT').count()


# In[14]:


dw_trn_alertas.select('CONT_RUT').distinct().count()


# ## Direccion regional

# In[15]:


df_negocios_nom = spark.table("DW.DW_TRN_NEGOCIOS_E")
df_unidad_operativa  = spark.table("DW.DW_DIM_UNIDAD_OPERATIVA")


# In[16]:


df_negocios_nom.columns


# In[17]:


df_negocios_nom.createOrReplaceTempView("DW_TRN_NEGOCIOS_NOM")
df_unidad_operativa.createOrReplaceTempView("DW_DIM_UNIDAD_OPERATIVA")
df_negocios_nom.columns


# In[18]:


#Verificamos para un rut en particular las entradas y fechas relacionadas. Escogesoo fecha de modificacion para encontra rle maximoregistro
spark.sql(' SELECT AUX.CONT_RUT, AUX.CONT_DV,NEGO_FECHA_CREACION_VO,NEGO_FECHA_IND_VERIFICACION_VO,NEGO_FECHA_INICIO_VO,NEGO_FECHA_MODIFICACION_VO,NEGO_FECHA_VIGENCIA FROM DW_TRN_NEGOCIOS_NOM AUX where CONT_RUT like "++DZAjc7cVaoNs7R%" ').show()


# In[19]:


# Consulta SQL
query = """
SELECT BL1.CONT_RUT, BL1.CONT_DV, BL1.UNOP_UNIDAD as UNOP_UNIDAD_I, BL1.UNOP_UNIDAD_GRAN_CONT, 
       BL2.UNOP_UNIDAD, 
       BL2.UNOP_COD_REGIONAL AS UNOP_COD_REGIONAL_I, 
       (CASE  
          WHEN CAST(CASE  
              WHEN BL1.UNOP_UNIDAD_GRAN_CONT IS NULL OR BL1.UNOP_UNIDAD_GRAN_CONT = '-9999'
              THEN BL1.UNOP_UNIDAD
              ELSE BL1.UNOP_UNIDAD_GRAN_CONT
              END AS FLOAT) >= 17000
          AND 
          CAST(CASE  
              WHEN BL1.UNOP_UNIDAD_GRAN_CONT IS NULL OR BL1.UNOP_UNIDAD_GRAN_CONT = '-9999'
              THEN BL1.UNOP_UNIDAD
              ELSE BL1.UNOP_UNIDAD_GRAN_CONT
              END AS FLOAT) < 18000
          THEN 17
          ELSE BL2.UNOP_COD_REGIONAL
       END) AS UNOP_COD_REGIONAL,
       BL1.NEGO_IND_EXPORTADOR_VO,
       BL1.NEGO_IND_PRIMERA_EXP_VO,
       BL1.NEGO_IND_VERIFICACION_VO,
       BL1.NEGO_NRO_FACTURAS_6MESES_VO,
       BL1.NEGO_NRO_FACTURAS_6MESES_VO
       
FROM (
   SELECT AL2.CONT_RUT, AL2.CONT_DV, AL2.UNOP_UNIDAD, AL2.UNOP_UNIDAD_GRAN_CONT,AL2.NEGO_IND_EXPORTADOR_VO,AL2.NEGO_IND_PRIMERA_EXP_VO,AL2.NEGO_IND_VERIFICACION_VO,AL2.NEGO_NRO_FACTURAS_6MESES_VO
   FROM (
      SELECT AUX.CONT_RUT, AUX.CONT_DV, MAX(NEGO_FECHA_VIGENCIA) as NEGO_FECHA_VIGENCIA
      FROM DW_TRN_NEGOCIOS_NOM AUX 
      GROUP BY AUX.CONT_RUT, AUX.CONT_DV
   ) AL1
   JOIN DW_TRN_NEGOCIOS_NOM AL2
   ON AL1.CONT_RUT = AL2.CONT_RUT AND AL1.NEGO_FECHA_VIGENCIA = AL2.NEGO_FECHA_VIGENCIA
) BL1
JOIN DW_DIM_UNIDAD_OPERATIVA BL2
ON BL1.UNOP_UNIDAD = BL2.UNOP_UNIDAD
"""
df_negocio = spark.sql(query).distinct()
### Ante la falta de una fecha de carga en el data warehouse se escoge el negocio con la fecha de vigencia mas reciente


# In[20]:


df_negocio.select('CONT_RUT').count()


# In[21]:


df_negocio.select('CONT_RUT').distinct().count()


# ### Tabla DW.DW_HEC_CONT_COMPORTAMIENTO_E 
# -----------------------------------------------------------------------------------------------------------------

# In[22]:


spark.sql('select CONT_RUT ,COCO_FECHA_CARGA_DW,COCO_AGNO_COMERCIAL from DW.DW_HEC_CONT_COMPORTAMIENTO_E where CONT_RUT like "MMjMmHvg0zm+0U9%" ').show()


# In[23]:


from pyspark.sql import Window
import pyspark.sql.functions as F

# Definir una ventana por CONT_RUT ordenada por COCO_AGNO_COMERCIAL descendente
window_spec = Window.partitionBy("CONT_RUT").orderBy(F.desc("COCO_AGNO_COMERCIAL"))

# Seleccionar los datos de la tabla con la fila de mayor COCO_AGNO_COMERCIAL para cada CONT_RUT
comportamiento = spark.sql("""
SELECT 
    CONT_RUT,
    CONT_DV,
    COCO_IMP_VENTAS_IVA,
    COCO_IMP_VENTAS_TRANSPORTE,
    COCO_MCA_1_CATEGORIA,
    COCO_MCA_2_CATEGORIA,
    COCO_MCA_AFECTO_IMPTO_ADIC,
    COCO_MCA_AFECTO_IMPTO_UNICO,
    COCO_MCA_DOBLE_DECL_F22,
    COCO_MCA_DONACIONES_CULTURALES,
    COCO_MCA_DONACIONES_DEPORTIVAS,
    COCO_MCA_DONACIONES_EDUCACIONALES,
    COCO_MCA_DONACIONES_POLITICAS,
    COCO_MCA_DONACIONES_UNIVERSIDAD,
    COCO_MCA_ES_EMPRESA,
    COCO_MCA_ES_GRAN_CONT,
    COCO_MCA_ES_MINERA,
    COCO_MCA_GLOBAL_COMPLE,
    COCO_MCA_IMP_PPM_FONDO_MUTUO,
    COCO_MCA_IMP_SOC_PROC,
    COCO_MCA_SIN_CLAS_IMP,
    COCO_MCA_TIPO_IMP,
    COCO_MTO_DEV_SOLICITADA_F22,
    COCO_MTO_VENTAS,
    TICO_SUB_TPO_CONTR,
    TRRE_COD_TMO_RTA,
    TRVE_COD_TMO_VTA,
    UNOP_UNIDAD_GRAN_CONT as UNIDAD_GRAN_CONTRIBUYENTE_COMPORTAMIENTO,
    COMU_COD_COMUNA_PRINCIPAL,
    COCO_AGNO_COMERCIAL
FROM 
    DW.DW_HEC_CONT_COMPORTAMIENTO_E
""")

# Aplicar la ventana y filtrar solo la fila con el mayor COCO_AGNO_COMERCIAL para cada CONT_RUT
df_comportamiento = comportamiento.withColumn(
    "row_number", 
    F.row_number().over(window_spec)
).filter(F.col("row_number") == 1).drop("row_number")

#count=87547893
#df_comportamiento=comportamiento.sample(withReplacement=False, fraction=0.1).limit(3)


# ### Tabla Actividad económica principal
# -----------------------------------------------------------------------------------------------------------------

# In[24]:


df_hec_cont = spark.table("DW.DW_HEC_CONT_COMPORTAMIENTO_E")
df_dim_actividad = spark.table("DW.DW_DIM_ACTIVIDAD_ECONOMICA_E")
df_trn_actividad = spark.table("DW.DW_TRN_ACTIVIDAD_ECONOMICA_E")

# Realizar JOINs
joined_df = df_hec_cont.alias("AL1") \
    .join(df_trn_actividad.alias("AL3"), 
          (F.col("AL1.CONT_RUT") == F.col("AL3.CONT_RUT")) & 
          (F.col("AL1.CONT_DV") == F.col("AL3.CONT_DV"))) \
    .join(df_dim_actividad.alias("AL2"), 
          F.col("AL1.ACEC_COD_ACTECO_PRINCIPAL") == F.col("AL2.ACEC_COD_ACTECO")) \
    .filter((F.col("AL1.ACEC_COD_ACTECO_PRINCIPAL") == F.col("AL3.ACTECO_COD_ACTECO")) & 
            (F.col("AL1.PERI_AGNO_TRIBUTARIO_RENTA") == 202300) & 
            (F.col("AL3.ACTECO_VIGENCIA") == 'S'))


acteco_principal = joined_df.select(
    F.col("AL1.CONT_RUT"),
    F.col("AL1.CONT_DV"),
#    F.col("AL1.ACEC_COD_ACTECO_PRINCIPAL"),
    F.col("AL2.ACEC_DES_ACTECO").alias("ACEC_DES_ACTECO_PPAL"),
#    F.col("AL2.ACEC_COD_RUBRO").alias("ACEC_COD_RUBRO_AP"),
    F.col("AL2.ACEC_DES_RUBRO").alias("ACEC_DES_RUBRO_PPAL"),
#    F.col("AL2.ACEC_COD_SUBRUBRO").alias("ACEC_COD_SUBRUBRO_AP"),
    F.col("AL2.ACEC_DES_SUBRUBRO").alias("ACEC_DES_SUBRUBRO_PPAL")
).distinct()


#acteco_principal=acteco_principal.sample(withReplacement=False, fraction=0.1).limit(3)
acteco_principal.show()


# In[25]:


acteco_principal.select('CONT_RUT').count()


# In[26]:


acteco_principal.select('CONT_RUT').distinct().count()


# ## Cruce de tabla de segmentos con alertas

# In[27]:


# Realizar el full outer join entre df_sgm_trn_final y dw_trn_alertas
result = df_sgm_trn_final.join(
    dw_trn_alertas, 
    on=["CONT_RUT", "CONT_DV"], 
    how="full_outer"
)

# Seleccionar todas las columnas de ambas tablas y asegurar que CONT_RUT y CONT_DV aparezcan una sola vez
result = result.select("CONT_RUT", "CONT_DV", 
                       "INICIO_SEGMENTO", 
                       "ES_EMPRESA", 
                       "ES_PERSONA", 
                       "Alerta_1019", 
                       "Alerta_2250", 
                       "Alerta_400X", 
                       "Alerta_4110", 
                       "Alerta_4111", 
                       "Alerta_4112", 
                       "Alerta_4113", 
                       "Alerta_52", 
                       "Alerta_5201", 
                       "Alerta_5203", 
                       "Alerta_53", 
                       "Alerta_5301")



# ### Cruce con la tabla de alertas

# In[28]:


# Realizar el full outer join entre df_sgm_trn_final y dw_trn_alertas
result = df_sgm_trn_final.join(
    dw_trn_alertas, 
    on=["CONT_RUT", "CONT_DV"], 
    how="full_outer"
)

# Seleccionar todas las columnas de ambas tablas y asegurar que CONT_RUT y CONT_DV aparezcan una sola vez
result = result.select("CONT_RUT", "CONT_DV", 
                       "INICIO_SEGMENTO", 
                       "ES_EMPRESA", 
                       "ES_PERSONA", 
                       "Alerta_1019", 
                       "Alerta_2250", 
                       "Alerta_400X", 
                       "Alerta_4110", 
                       "Alerta_4111", 
                       "Alerta_4112", 
                       "Alerta_4113", 
                       "Alerta_52", 
                       "Alerta_5201", 
                       "Alerta_5203", 
                       "Alerta_53", 
                       "Alerta_5301")


# ### Cruce con direccion regional yotras variables asociado al ultimo negocio

# In[29]:


# Realizar el full outer join entre el resultado y df_negocio
result_final = result.join(
    df_negocio,
    on=["CONT_RUT", "CONT_DV"],
    how="full_outer"
)

# Seleccionar todas las columnas deseadas
result_final = result_final.select(
    "CONT_RUT", 
    "CONT_DV", 
    "INICIO_SEGMENTO", 
    "ES_EMPRESA", 
    "ES_PERSONA", 
    "Alerta_1019", 
    "Alerta_2250", 
    "Alerta_400X", 
    "Alerta_4110", 
    "Alerta_4111", 
    "Alerta_4112", 
    "Alerta_4113", 
    "Alerta_52", 
    "Alerta_5201", 
    "Alerta_5203", 
    "Alerta_53", 
    "Alerta_5301", 
    "UNOP_UNIDAD_GRAN_CONT", 
    "UNOP_UNIDAD", 
    "UNOP_COD_REGIONAL", 
    "NEGO_IND_EXPORTADOR_VO", 
    "NEGO_IND_PRIMERA_EXP_VO", 
    "NEGO_IND_VERIFICACION_VO", 
    "NEGO_NRO_FACTURAS_6MESES_VO"
)



# In[30]:


result_final.columns


# In[31]:


df_comportamiento.columns


# ### Cruce con indicadores de comportamiento

# In[32]:


# Realizar el full outer join entre el resultado final y df_comportamiento
result_final_comportamiento = result_final.join(
    df_comportamiento,
    on=["CONT_RUT", "CONT_DV"],
    how="full_outer"
)

# Seleccionar todas las columnas deseadas
result_final_comportamiento = result_final_comportamiento.select(
    "CONT_RUT", 
    "CONT_DV", 
    "INICIO_SEGMENTO", 
    "ES_EMPRESA", 
    "ES_PERSONA", 
    "Alerta_1019", 
    "Alerta_2250", 
    "Alerta_400X", 
    "Alerta_4110", 
    "Alerta_4111", 
    "Alerta_4112", 
    "Alerta_4113", 
    "Alerta_52", 
    "Alerta_5201", 
    "Alerta_5203", 
    "Alerta_53", 
    "Alerta_5301", 
    "UNOP_UNIDAD_GRAN_CONT", 
    "UNOP_COD_REGIONAL", 
    "NEGO_IND_EXPORTADOR_VO", 
    "NEGO_IND_PRIMERA_EXP_VO", 
    "NEGO_IND_VERIFICACION_VO", 
    "NEGO_NRO_FACTURAS_6MESES_VO",
    "COCO_IMP_VENTAS_IVA", 
    "COCO_IMP_VENTAS_TRANSPORTE", 
    "COCO_MCA_1_CATEGORIA", 
    "COCO_MCA_2_CATEGORIA", 
    "COCO_MCA_AFECTO_IMPTO_ADIC", 
    "COCO_MCA_AFECTO_IMPTO_UNICO", 
    "COCO_MCA_DOBLE_DECL_F22", 
    "COCO_MCA_DONACIONES_CULTURALES", 
    "COCO_MCA_DONACIONES_DEPORTIVAS", 
    "COCO_MCA_DONACIONES_EDUCACIONALES", 
    "COCO_MCA_DONACIONES_POLITICAS", 
    "COCO_MCA_DONACIONES_UNIVERSIDAD", 
    "COCO_MCA_ES_EMPRESA", 
    "COCO_MCA_ES_GRAN_CONT", 
    "COCO_MCA_ES_MINERA", 
    "COCO_MCA_GLOBAL_COMPLE", 
    "COCO_MCA_IMP_PPM_FONDO_MUTUO", 
    "COCO_MCA_IMP_SOC_PROC", 
    "COCO_MCA_SIN_CLAS_IMP", 
    "COCO_MCA_TIPO_IMP", 
    "COCO_MTO_DEV_SOLICITADA_F22", 
    "COCO_MTO_VENTAS", 
    "TICO_SUB_TPO_CONTR", 
    "TRRE_COD_TMO_RTA", 
    "TRVE_COD_TMO_VTA", 
    "COMU_COD_COMUNA_PRINCIPAL",
    "UNIDAD_GRAN_CONTRIBUYENTE_COMPORTAMIENTO"
)


# In[33]:


result_final_comportamiento.columns


# ### Cruce con  actividad economica principal

# In[34]:


# Realizar el full outer join entre el resultado final y acteco_principal
result_final_acteco = result_final_comportamiento.join(
    acteco_principal,
    on=["CONT_RUT", "CONT_DV"],
    how="full_outer"
)

# Seleccionar todas las columnas deseadas
result_final_acteco = result_final_acteco.select(
    "CONT_RUT", 
    "CONT_DV", 
    "INICIO_SEGMENTO", 
    "ES_EMPRESA", 
    "ES_PERSONA", 
    "Alerta_1019", 
    "Alerta_2250", 
    "Alerta_400X", 
    "Alerta_4110", 
    "Alerta_4111", 
    "Alerta_4112", 
    "Alerta_4113", 
    "Alerta_52", 
    "Alerta_5201", 
    "Alerta_5203", 
    "Alerta_53", 
    "Alerta_5301", 
    "UNOP_UNIDAD_GRAN_CONT", 
    "UNOP_COD_REGIONAL", 
    "NEGO_IND_EXPORTADOR_VO", 
    "NEGO_IND_PRIMERA_EXP_VO", 
    "NEGO_IND_VERIFICACION_VO", 
    "NEGO_NRO_FACTURAS_6MESES_VO", 
    "COCO_IMP_VENTAS_IVA", 
    "COCO_IMP_VENTAS_TRANSPORTE", 
    "COCO_MCA_1_CATEGORIA", 
    "COCO_MCA_2_CATEGORIA", 
    "COCO_MCA_AFECTO_IMPTO_ADIC", 
    "COCO_MCA_AFECTO_IMPTO_UNICO", 
    "COCO_MCA_DOBLE_DECL_F22", 
    "COCO_MCA_DONACIONES_CULTURALES", 
    "COCO_MCA_DONACIONES_DEPORTIVAS", 
    "COCO_MCA_DONACIONES_EDUCACIONALES", 
    "COCO_MCA_DONACIONES_POLITICAS", 
    "COCO_MCA_DONACIONES_UNIVERSIDAD", 
    "COCO_MCA_ES_EMPRESA", 
    "COCO_MCA_ES_GRAN_CONT", 
    "COCO_MCA_ES_MINERA", 
    "COCO_MCA_GLOBAL_COMPLE", 
    "COCO_MCA_IMP_PPM_FONDO_MUTUO", 
    "COCO_MCA_IMP_SOC_PROC", 
    "COCO_MCA_SIN_CLAS_IMP", 
    "COCO_MCA_TIPO_IMP", 
    "COCO_MTO_DEV_SOLICITADA_F22", 
    "COCO_MTO_VENTAS", 
    "TICO_SUB_TPO_CONTR", 
    "TRRE_COD_TMO_RTA", 
    "TRVE_COD_TMO_VTA", 
    "COMU_COD_COMUNA_PRINCIPAL", 
    "ACEC_DES_ACTECO_PPAL", 
    "ACEC_DES_RUBRO_PPAL", 
    "ACEC_DES_SUBRUBRO_PPAL",
    "UNIDAD_GRAN_CONTRIBUYENTE_COMPORTAMIENTO"
).distinct()


# In[35]:


result_final_acteco.select('CONT_RUT').count()


# In[36]:


result_final_acteco.select('CONT_RUT').distinct().count()


# In[37]:


# Consideraremos solo los registros que tienen CONT_RUT no repetidos. Esto se hace debido a que no hay unicidad de registros por la naturaleza de ciertas tablas.

# 1. Contar las ocurrencias de CONT_RUT
conteo_rut = result_final_acteco.groupBy("CONT_RUT").count()

# 2. Filtrar los RUT que aparecen una vez
rut_unico = conteo_rut.filter(conteo_rut["count"] == 1)

# 3. Unir de nuevo con el DataFrame original para obtener las filas correspondientes
resultado_final = result_final_acteco.join(rut_unico.select("CONT_RUT"), on="CONT_RUT", how="inner")

# 4. Seleccionar solo las columnas del DataFrame original
resultado_final = resultado_final.select(result_final_acteco.columns)


# In[38]:


# Se guarda el archivo final en el datalake. 
result_final.write.mode('overwrite').format("parquet").save("abfs://data@datalakesii.dfs.core.windows.net/DatosOrigen/lr-629/APA/Analisis_factura/data_contribuyentes")


# In[39]:


spark.stop()

