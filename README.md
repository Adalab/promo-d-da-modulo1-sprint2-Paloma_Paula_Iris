# promo-d-da-modulo3-sprint1-Paloma_Paula_Iris

# Repositorio con los ejercicios de Pair Programming del Módulo 3, Sprint 1 del equipo1.

### Librerías:
Para la realización de estos modelos de machine learning se han utilizado varias librerías (adjuntamos su documentación);
- Pandas: https://pandas.pydata.org/docs/
- Numpy: https://numpy.org/
- Matplotlib: https://matplotlib.org/stable/index.html
- Seaborn: https://seaborn.pydata.org/
- Sklearn: https://scikit-learn.org/stable/
- Statmodels: https://www.statsmodels.org/stable/index.html
- Spicy: https://docs.scipy.org/doc/scipy/
- Sidetable: https://pypi.org/project/sidetable/
- Sys: https://docs.python.org/3/library/sys.html
- Imblearn: https://imbalanced-learn.org/stable/

### Herramientas utilizadas: 
- Visual Studio Code.

### Organización del Repositorio:

En este repositorio se encuentran dos carpetas principales con sus respectivos archivos; una para el modelo de regresión lineal y otra para el de regresión logística y una tercera para incluir los archivos de python con funciones urilizadas:

1. Regresion_lineal: carpeta para el desarrollo de un modelo predictivo con regresión lineal.
    - archivos: documento csv con el archivo a predecir más los archivos generados posteriormente en formato pickle para proseguir con el modelo.
    - graficas: algunas imágenes .png para ilustrar el poceso de EDA.
    - r_lineal: cuadernos .jpynb para la realización y comentario del modelo predictivo.

2. Regresion_logistica: carpeta para el desarrollo de un modelo predictivo con regresión logística.
    - archivos: documento csv con el archivo a predecir más los archivos generados posteriormente en formato pickle para proseguir con el modelo.
    - graficas: algunas imágenes .png para ilustrar el poceso de EDA.
    - rlogistica: cuadernos .jpynb para la realización y comentario del modelo predictivo.

### Modelos de Machine Learning:
1. Regresión Lineal:

Para nuestro proyecto de regresión lineal hemos elegido un dataset referido al precio de distintos productos y servicios en diferentes países y ciudades (https://www.kaggle.com/datasets/mvieira101/global-cost-of-living)
- Después del EDA del dataset decidimos tomar como variable respuesta ‘Basic’, columna referida al precio de los servicios básicos (luz, gas, agua) debido al interés que toman estos datos para la vida cotidiana, relevante para proceder a un análisis más profundo y comparativo del coste de vida medio entre países con respecto a las necesidades básicas.

Se ha realizado el procedimiento y tratamiento necesario de los datos tanto de esta variable respuesta como de las demás variables, las predictoras.
- Tras un breve análisis estadístico percibimos variables que son redundantes mediante la correlación entre ellas, por lo que decidimos eliminarlas.

Inicalemnte valoramos a realizar un modelo de regresión lineal, para lo que comprobamos si nuestro conjunto de datos cumple con las asunciones necesarias para llevar a cabo este tipo de modelo.
- Confirmamos que no se cumple la normalidad en la variable respuesta y además no se puede normalizar (método boxcox de normalización de variables).
- Las variables predictoras tampoco cumplen con las asunciones de homocedasticidad ni interdependencia.

Con propósitos académicos tenemos que emplear este tipo de regresión, aunque en un primer momento no escogeríamos este tipo de regresión

Para poder realizar los modelos predictivos, preparamos los datos:
- Para las variables numéricas la estandarización con un Robust Scaler porque el dataset tiene outliers
- Para la única variable categórica probamos diferentes codificaciones, proporcionando un tratamiento ordinal y nominal con propósitos de prueba.

Tanteamos los diferentes modelos:
a. Regresión lineal
b. Decision Tree
c. Random Forest

NEXT STEPS:
- Mejorar los hiperparámetros del Random Forest, porque es el que mejores métricas ha obtenido
- Probar diferentes combinaciones de encoding
- Distinto tratamiento de outliers
- Plantear otras variables predictoras o enfocar el proyecto con otra variable respuesta
-----------------------------------------------------------------------------------------------------------------------------------------

2. Regresión Logística:
Para nuestro proyecto de regresión logística hemos elegido un dataset que registra la permanencia de clientes en una compañía telefónica según diferentes factores. (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Después del EDA del dataset decidimos tomar como variable respuesta ‘Churn’, columna referida a la permanencia del cliente debido al interés que toman estos datos para la compañía, relevante para proceder a un análisis sobre aquellos factores que puedan influir en la decisión que toma el cliente de abandonar o no la compañía.

Se ha realizado el procedimiento y tratamiento necesario de los datos tanto de esta variable respuesta como de las demás variables, las predictoras.
- Eliminamos variables redundantes y pasamos a la estandarización de las variables numéricas (Standard Scaler porque no tenemos outliers) y hacemos el encoding de las variables categóricas, diferenciándolas entre nominales (get_dummies) y ordinales, fijándonos en sus pesos (map)
- En cuanto al balanceo de la variable respuesta optamos por probar a realizar modelos con y sin balancear. El balanceo lo hemos hecho con RandomUnderSampler.

Tanteamos los diferentes modelos:
a. Regresión logística
b. Decision Tree
c. Random Forest

NEXT STEPS:
- Mejorar los hiperparámetros del Random Forest, que de nuevo es el que mejores métricas ha obtenido
- Probar diferentes combinaciones de encoding
- Probar otros métodos de balanceo