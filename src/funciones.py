# IMPORTACIONES
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
#=========================


# FUNCIONES EDA:
#============================
def exploracion(dataframe):
    '''
    Función que nos permite explorar varios aspectos de un dataframe mostrando
    unas filas, el numero de filas y columnas, los tipos de datos, la cantidad de nulos por columna,
    la cantidad de filas duplicadas, y los principales estadísticos.
        Parametros:
                dataframe (dataframe): un dataframe que queremos explorar
                nombre (string): un titulo para el dataframe
    '''
    print(".............................................")
    print(f"En el dataframe hay {dataframe.shape[0]} filas y {dataframe.shape[1]} columnas.")
    print(".............................................")    
    print(f"5 filas aleatorias:")
    display(dataframe.sample(5))
    print(".............................................")
    print(f"Los tipos de data por columna del dataframe son:")
    display(dataframe.dtypes.reset_index().T)
    print(".............................................")
    print(f"La cantidad de nulos por columna del dataframe son:")
    display(dataframe.isnull().sum().reset_index().T)
    print(".............................................")
    print(f"El porcentaje de nulos por columna del dataframe son:")
    display(dataframe.isnull().sum() * 100 / dataframe.shape[0])
    print(".............................................")
    if dataframe.duplicated().sum() != 0:
        print(f"En el dataframe hay {dataframe.duplicated().sum()} filas duplicadas.")
    else:
        print(f"No hay filas duplicadas.")
    print(".............................................")
    print(f"Los principales estadísticos de las columnas numéricas son:")
    display(dataframe.describe())
    print(".............................................")
    print(f"Los principales estadísticos de las columnas categóricas son:")
    display(dataframe.describe(include=object))
    print(".............................................")


def detectar_outliers(lista_columnas, dataframe): 
    '''
    Función que extrae los indices de los outliers para cada una de las columnas del dataframe y devuelve
    un diccionario con los nombres de las columnas como keys y los indices de los outliers como values.
        Parametros:
                lista_columnas: lista con los nombres de las columnas (numéricas) de las cuales queremos detectar los outliers
                dataframe (dataframe): un dataframe que queremos explorar
    '''
    dicc_indices = {} 

    for col in lista_columnas:
        
        #calculamos los cuartiles Q1 y Q3
        Q1 = np.nanpercentile(dataframe[col], 25)
        Q3 = np.nanpercentile(dataframe[col], 75)
        
        # calculamos el rango intercuartil
        IQR = Q3 - Q1
        
        # calculamos los límites
        outlier_step = 1.5 * IQR
        
        # filtramos nuestro dataframe para indentificar los outliers
        outliers_data = dataframe[(dataframe[col] < Q1 - outlier_step) | (dataframe[col] > Q3 + outlier_step)]
        
        if outliers_data.shape[0] > 0:
            dicc_indices[col] = (list(outliers_data.index))
    
    return dicc_indices 


# FUNCIONES TEST-ESTADÍSTICOS:
#===========================
def normalidad(dataframe):
    '''
    Función que utiliza el test de saphiro para calcular si una variable sigue una distribución normal y devuelve
    una respuesta en formato texto junto con el pvalue
        Parametros:
                dataframe (dataframe): un dataframe que queremos explorar. Las variables deben ser numércias.
    '''
    dist_variables = []
    for col in dataframe.columns:
        if stats.shapiro(dataframe[col])[1] > 0.05:
            text1 = f'{col} SÍ tiene distribución normal -> {stats.shapiro(dataframe[col])[1]}'
            dist_variables.append(text1)
        else:
            text2 =  f'{col}, NO tiene distribución normal -> {stats.shapiro(dataframe[col])[1]}'
            dist_variables.append(text2)
    return dist_variables


# FUNCIONES METRICAS ML
#===========================
def metricas(y_test, y_train, y_test_pred, y_train_pred, tipo_modelo):
    
    resultados = {'MAE': [mean_absolute_error(y_test, y_test_pred), mean_absolute_error(y_train, y_train_pred)],
                'MSE': [mean_squared_error(y_test, y_test_pred), mean_squared_error(y_train, y_train_pred)],
                'RMSE': [np.sqrt(mean_squared_error(y_test, y_test_pred)), np.sqrt(mean_squared_error(y_train, y_train_pred))],
                'R2':  [r2_score(y_test, y_test_pred), r2_score(y_train, y_train_pred)],
                 "set": ["test", "train"]}
    df = pd.DataFrame(resultados)
    df["modelo"] = tipo_modelo
    return df