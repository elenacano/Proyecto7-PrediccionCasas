# Proyecto 7: Estimación de precios de alquiler de viviendas en la Comunidad de Madrid

<img src="imagenes/portada3.webp" width="700" height="500">


La estimación de los precios inmobiliarios es un área crucial que combina el análisis de negocios con la ciencia de datos. En este proyecto, se abordará el reto de prever el precio de alquiler de viviendas. El conjunto de datos que utilizarás proviene del mercado inmobiliario de Madrid y contiene información detallada sobre las propiedades, como su superficie, ubicación, número de habitaciones, tipo de inmueble, entre otros aspectos relevantes.


## Objetivos

1. **Preprocesamiento**: Abarca todas las etapas de preparación de los datos: EDA, gestión de nulos, encoding, outliers y estandarización.

2. **Modelos predictivos**: Selección y prueba de los modelos más precisos usando Scikitlearn.

3. **Presentación de los datos**: Utilizar Streamlit como plataforma para la consulta sencilla de las predicciones.


## Estructura del repositorio

El proyecto está construido de la siguiente manera:

Hay dos carpetas `Modelo1` y `Modelo2` que difieren en el tratamiento de los datos para obtener distintos resultados en los modelos de predicción. Dinalmente se ha usado el Modelo1.

Ambas carpetas siguen una misma estructura:

- **datos/**: Carpeta que contiene archivos `.csv` o `.pkl` generados durante la captura y tratamiento de los datos.

- **notebooks/**: Carpeta que contiene los archivos `.ipynb` utilizados en la captura y tratamiento de los datos. Están numerados para su ejecución secuencial.
  - `1-eda-nulos`
  - `2-encoding`
  - `3-outliers`
  - `4-estandarizacion.`
  - `5-modelo`
  - `6-prediccion`

- **src/**: Carpeta que contiene los archivos `.py`, con las funciones y variables utilizadas en los distintos notebooks.

- `.gitignore`: Archivo que contiene los archivos y extensiones que no se subirán a nuestro repositorio, como los archivos .env, que contienen contraseñas.


## Lenguaje, librerías y temporalidad
El proyecto fué elaborado con Python 3.9 y múltiples librerías de soporte:

- [Pandas](https://pandas.pydata.org/docs/)
- [Numpy](https://numpy.org/doc/)
- [Seaborn](https://seaborn.pydata.org)
- [Matplotlib](https://matplotlib.org/stable/index.html)
- [streamlit](https://docs.streamlit.io)
- [scikitlearn](https://scikit-learn.org/stable/)
- [itertools](https://docs.python.org/3/library/itertools.html)
- [warnings](https://docs.python.org/3/library/warnings.html)


Este proyecto es funcional a fecha 24 de noviembre de 2024.


## Instalación

1. Clona el repositorio
   ```sh
   git clone git@github.com:elenacano/Proyecto7-PrediccionCasas.git
   ```

2. Instala las librerías que aparecen en el apartado anterior. Utiliza en tu notebook de Jupyter:
   ```sh
   pip install nombre_librería
   ```

3. Cambia la URL del repositorio remoto para evitar cambios al original.
   ```sh
   git remote set-url origin usuario_github/nombre_repositorio
   git remote -v # Confirma los cambios
   ```

4. Ejecuta el código de los notebooks en el orden especificado, modificándolo si es necesario.

