# Reid-Stand

## Установка

1. Miniconda:

Команды запускать из домашней директории
```
$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod +x Miniconda3-latest-Linux-x86_64.sh
$ ./Miniconda-latest-Linux-x86_64.sh
```
На все вопросы установочника отвечать `yes`
После установки необходимо перезапустить терминал. В случае успешной установки перед именем пользователя появится надпись `(base)`

2. torchreid
```
$ git clone https://github.com/erafier/deep-person-reid.git
$ cd deep-person-reid/
$ conda create --name reid python=3.7
$ conda activate reid
$ pip install -r requirements.txt
```
Установка на машину с гпу: `$ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`  
Установка без гпу: `$ conda install pytorch torchvision cpuonly -c pytorch`

`
$ pip install pillow==6.2.2
$ python setup.py develop`

Все дальнейшие действия и запуск программы должны осуществляться из окружения reid. Перед именем пользователя в терминале должно быть написано (reid)

3. Установить необходимые зависимости окружении reid:

- flask: `pip install flask`
- pycocotools: `pip install cython; pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`

4. Скачиваем данный репозирорий командой

`git clone https://github.com/yaros1avsv/Reid-Stand.git`

5. Преобразование YoloV4 для использования в TensorFlow
