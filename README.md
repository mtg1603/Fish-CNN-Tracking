# Fish CNN Tracking
Sistema de detecção e rastreamento de peixes para análises comportamentais utilizando a rede neural convolucional MobileNetV2 para a detecção e a biblioteca dlib para o rastreamento.

Os arquivos originais de pesos, classe e configuração encontram-se na página de modelos do framework [TensorFlow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) e também podem ser aplicados de maneira genérica no sistema (com o intuito de testes). Os treinados, voltados para a detecção de peixes, podem ser baixados [aqui](https://drive.google.com/file/d/1_kXgUS5gyzi21rI3urcUcWVZlSbG-LSU/view?usp=sharing).

As seguintes versões dos programas e distribuições foram utilizadas para a execução do algoritmo:
- Python 3.6.9
- TensorFlow 1.15.2
- OpenCV 4.2.0
- dlib 19.19.0
- CUDA 10.0
- cuDNN 7.6.3
- Ubuntu 18.04
- JetPack 4.3

Agradeço à todos os desenvolvedores que disponibilizaram as ferramentas para o desenvolvimento deste sistema.

**Novas funcionalidades foram implementadas no sistema, como a escolha e utilização de mais detectores (YOLOv3 e YOLOv3-tiny) e rastreadores (MOSSE, KCF e CSRT).**
