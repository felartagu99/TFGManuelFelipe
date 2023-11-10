# Sistema HW/SW para generar datasets de objetos físicos para entrenar CNN

**Descripción:** Se propone diseñar un sistema HW/SW para la elaboración de un dataset para el posterior entrenamiento de una red neuronal convolucional. Este sistema deberá comprender una parte hardware, la cual debe tener una cámara para tomar las fotografías y una parte software para realizar el proceso de captación de imágenes y etiquetado. Con dicho software, se hará una clasificación manual según categoría, por parte del usuario usando el software desarrollado.

**Objetivos:** El objetivo de este TFG es realizar una aplicación universal para realizar datasets de objetos físicos.

**Tutor:** Jose Antonio Diaz Navarro

**Cotutor:** Salvador Canas Moreno

**Departamento:** Arquitectura y Tecnología de Computadores

# Segment Anything Model con Label Studio ML Backend

## Requisitos previos
Para este ejemplo, recomendamos utilizar Docker para alojar el SAM ML Backend y Label Studio. Docker facilita la instalación del software sin necesidad de otros requisitos del sistema y ayuda a que el proceso de instalación y mantenimiento sea mucho más manejable. Para usuarios de escritorio o portátiles, la forma más rápida de obtener Docker es instalando el cliente oficial Docker Desktop para sistemas operativos Mac y Windows o instalando Docker mediante el administrador de paquetes oficial para sistemas Linux.

El Modelo Segment Anything es un modelo base grande y complejo que funciona mejor en una GPU. Dado que muchas personas probarán este software en hardware común como portátiles o computadoras de escritorio, por defecto, el modelo se envía con Mobile SAM habilitado. El backend detectará automáticamente si tienes una GPU disponible, utilizando el hardware más adecuado para tu sistema.

Consulta la documentación oficial de Docker y la documentación de Docker Compose para habilitar el paso de GPU para tus contenedores invitados.

Como mínimo, tu sistema debe tener 16 GB de RAM disponible, con al menos 8 GB asignados al tiempo de ejecución de Docker.

También debes tener Git instalado en tu sistema para descargar el repositorio del backend de Label Studio ML.

## Clonar el Repositorio
Después de instalar Docker y Git, el siguiente paso es clonar el repositorio git del backend de Label Studio en tu sistema.

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
```
## Cambiar al directorio de trabajo de SAM
```bash
cd label-studio-ml-backend/label_studio_ml/examples/segment_anything_model
```

## Crear la imagen de Docker
```bash 
docker build . -t sam:latest 
```
En verdad puedes ponerle cualquier nombre.

### Debemos verificar que la imagen se ha creado, debe aparecer algo parecido a esto:
```bash 
REPOSITORY                 TAG       IMAGE ID       CREATED       SIZE
humansignal/sam            v0        5f49434f8a86   2 weeks ago   4.62GB
sam                        latest    0bcd2a66fd4e   2 weeks ago   4.61GB
```
# Usar el SAM ML Backend
Con la imagen construida, es hora de crear un proyecto de segmentación de imágenes utilizando Label Studio.

## Instalar Label Studio 

Primero, debes instalar Label Studio. Para este ejemplo, el SAM ML Backend depende de habilitar el servicio de almacenamiento local. Para iniciar una instancia de Label Studio con esto habilitado, ingresa el siguiente comando:

```bash
docker run -it -p 8080:8080 \
    -v $(pwd)/mydata:/label-studio/data \
    --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
    --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/images \
    heartexlabs/label-studio:latest
```
Este comando le indica a Docker que lance Label Studio, lo haga disponible para ver en http://localhost:8080, almacene la base de datos y los archivos de tarea en tu disco duro local y habilite el servicio de archivos locales. Una vez que Label Studio se haya iniciado, puedes acceder con tu navegador a http://localhost:8080, donde se te presentará la pantalla de inicio de sesión de Label Studio.

# Problemas Mac

**Portátil con estas carecterísticas:**

```bash
Chip: Apple M1 Pro
Memoria: 16 GB 
Disco de Arranque: Macintosh HD
macOS: Sonoma 14.1.1
``````
Único problema que tuvimos durante la instalación fue resuelto en el siguiente enlace: 

```bash
https://github.com/docker/for-mac/issues/3785
``````

# Problemas Windows
**Portátil con estas carecterísticas:**