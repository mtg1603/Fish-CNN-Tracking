# Fish CNN Tracking
## EN-US

Fish detection and tracking system for behavioral analysis using convolutional neural networks for detection (YOLOv3, YOLOv3-fish or MobileNetV2) and correlational filters for tracking (dlib, CSRT, KCF and MOSSE). There are two versions of the algorithm: the desktop version and the NVIDIA Jetson Nano board version.

The original weights, class and configuration files can be found on the [TensorFlow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and [Darknet](https://github.com/AlexeyAB/darknet) framework's templates page. The generic models can be applied to the algorithm (for testing purposes). The fish trained models on the OIDv4 "fish" class can be downloaded [here](https://drive.google.com/file/d/1iQtRNgVBBtLTQAGEkm30cYjh7lC5Psby/view?usp=sharing).

A manually annotated dataset was created with 4 videos, in which 2 of them have inanimate fishes (toy) in an experimental environment and the others a real fishes in aquariums. This dataset was used to analyze the detection and tracking capability of the system. The DarkLabel tool was used for the manual annotation of the videos. The dataset can be downloaded [here](https://drive.google.com/file/d/1BPQ7L59aczTEcvJ9HMm4xwrsz5mL1GxC/view?usp=sharing).

The following dependecies and versions were used to run the algorithm:
- Python 3.6.9
- TensorFlow 1.15.2
- OpenCV 4.2.0
- dlib 19.19.0
- CUDA 10.0
- cuDNN 7.6.3
- PAPI 5.3.2
- tegrastats 2.2
- Ubuntu 18.04
- JetPack 4.3

Thanks to all developers who made the tools available for the development of this algorithm and fish enthusiasts for the videos.

## PT-BR

Sistema de detecção e rastreamento de peixes para análises comportamentais utilizando redes neurais convolucionais (YOLOv3, YOLOv3-fish ou MobileNetV2) para a detecção e filtros correlacionais (dlib, CSRT, KCF e MOSSE) para o rastreamento. Existem duas versões do algoritmo: uma para desktop e outra para a placa NVIDIA Jetson Nano.

Os arquivos originais de pesos, classe e configuração encontram-se na página de modelos do framework [TensorFlow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) e do [Darknet](https://github.com/AlexeyAB/darknet) e também podem ser aplicados de maneira genérica no sistema (com o intuito de testes). Os modelos treinados na classe "fish" do OIDv4 podem ser baixados [aqui](https://drive.google.com/file/d/1iQtRNgVBBtLTQAGEkm30cYjh7lC5Psby/view?usp=sharing).

Foi elaborado um dataset manualmente anotado com 4 vídeos, sendo 2 deles com peixes inanimados (brinquedos) em um ambiente experimental e os demais com peixes reais em aquários. Este dataset foi utilizado para avaliar a capacidade de detecção e rastreamento do sistema. Para realizar as anotações nos vídeos, a ferramenta DarkLabel foi utilizada. O dataset pode ser baixado [aqui](https://drive.google.com/file/d/1BPQ7L59aczTEcvJ9HMm4xwrsz5mL1GxC/view?usp=sharing).

As seguintes versões dos programas e distribuições foram utilizadas para a execução do algoritmo:
- Python 3.6.9
- TensorFlow 1.15.2
- OpenCV 4.2.0
- dlib 19.19.0
- CUDA 10.0
- cuDNN 7.6.3
- PAPI 5.3.2
- tegrastats 2.2
- Ubuntu 18.04
- JetPack 4.3

Agradeço à todos os desenvolvedores que disponibilizaram as ferramentas para o desenvolvimento deste sistema e aos entusiastas de peixes pelos vídeos.
