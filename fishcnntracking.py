# ⍺⍺⍺ ---- Rastreamento de Peixes ---- ⍺⍺⍺

# Importa os módulos necessários para o funcionamento do programa
from collections import deque # Módulo de estrutura de dados do tipo lista de indexação rápida
from imutils.video import VideoStream # Módulo para obtenção de vídeo em tempo real
from imutils.video import FPS # Módulo para obtenção dos frames por segundo de execução do vídeo
from object_detection.utils import visualization_utils as vis_util # Módulo de visualização da API de detecção de objetos do TensorFlow
from object_detection.utils import label_map_util # Módulo de manipulação de labels da API de detecção de objetos do TensorFlow
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np # Módulo de operações matemáticas
import pandas as pd # Módulo de manipulação e análise de dados
import dlib # Módulo contendo diversas bibliotecas para cálculos númericos, processamento de imagens e afins
import cv2 # Módulo de ferramentas para visão computacional (OpenCV)
import tensorflow as tf # Módulo de ferramentas de redes neurais (TensorFlow)
import imutils # Módulo de ferramentas para manipulação de imagem/vídeo
import time # Módulo de ferramentas de tempo
import sys # Módulo de ferramentas do sistema
import os # Módulo de ferramentas de interface do sistema

# Variável de inicialização do modelo do TensorFlow, utilizado para representar um fluxo de dados como um gráfico
modelo = tf.Graph()

# Menu de opções
print('\n⍺⍺⍺ ---- Rastreamento de Peixes ---- ⍺⍺⍺\n') # Introdução ao programa

# Opções referentes ao uso de GPU ou CPU
print('Deseja-se rodar utilizando a GPU ou a CPU?') # Mensagem ao usuário solicitando a escolha do processador que será utilizado na rede neural
print('Em grande parte dos casos, utilizar a GPU garante uma velocidade de processamento superior à CPU.') # Mensagem informando ao usuário o benefício da GPU
print('Para usar a GPU, é necessário ter instalado os drivers e pacotes CUDA (NVIDIA) ou ROCm (AMD/ATI) e ter uma placa compatível.') # Mensagem informando ao usuário o necessário para executar na GPU
escolhaproc = input('>') # Armazena a opção de processador escolhida

# Caso seja digitado corretamente CPU ou GPU
if escolhaproc == "GPU" or escolhaproc == "gpu" or escolhaproc == "CPU" or escolhaproc == "cpu":
	# Opções referentes à seleção da fonte de vídeo
	print('\nDeseja-se usar uma fonte de vídeo a partir de:') # Escolha da fonte de vídeo
	print('1 - Arquivo') # Fonte de vídeo a partir de um arquivo salvo/gravado
	print('2 - Tempo real') # Fonte de vídeo a partir de uma Webcam/USB/CSI
	opcaofonte = int(input('>')) # Armazena de fonte de vídeo a opção escolhida

# Caso seja digitado algo diferente, saia do programa
else:
	print('\nOpção inválida!\n') # Mensagem ao usuário dizendo que a opção digitada foi incorreta
	sys.exit() # Encerra o programa

# Tratamento da fonte de vídeo escolhida
if opcaofonte == 1: # Opção de vídeo a partir de um arquivo salvo/gravado
	print('\nDigite o endereço completo do arquivo de vídeo, incluindo seu nome e extensão:') # Mensagem ao usuário solicitando o local do arquivo de vídeo
	localfonte = input('>') # Recebe do usuário o local completo de onde está armazenado o arquivo de vídeo
	vs = cv2.VideoCapture(localfonte) # Armazena o arquivo de vídeo em uma variável
	fpscerto = vs.get(cv2.CAP_PROP_FPS) # Obtêm o valor de FPS do vídeo armazenado, para a geração correta do arquivo de vídeo com as informações de rastreamento
	fpscerto = round(fpscerto, 2) # Realiza o arredondamento do FPS obtido em 2 casas após a vírgula

	# Mensagem solicitando ao usuário o local que será armazenado a captura do vídeo exibido na tela em tempo real
	print('\nDigite o nome (sem a extensão) e o local do arquivo que deseja armazenar o vídeo com os pontos de rastreamento:')
	saidastream = input('>') # Recebe do usuário o local completo de onde será armazenado a captura do vídeo

elif opcaofonte == 2: # Opção de vídeo a partir de uma fonte Webcam/USB/CSI
	print('\nFontes de vídeo disponíveis para a captura:\n') # Mensagem ao usuário listando as fontes de vídeo presentes no dispositivo
	os.system("ls -l /dev/video*") # Lista as fontes de vídeo conectadas e disponíveis
	print('\nCaso encontre problemas com o dispositivo escolhido, talvez seja necessário utilizar ou aplicar outros parâmetros.') # Mensagem informativa sobre a necessidade de parâmtros adicionais
	print('\nDigite qual dispositivo será usado para a captura:') # Mensagem ao usuário solicitando a escolha do dispositivo de vídeo
	dispositivocap = input('>') # Armazena o dispositivo de vídeo escolhido pelo usuário em uma variável
	vs = cv2.VideoCapture(dispositivocap) # Armazena o vídeo em tempo real em uma variável
	print('\nFoi utilizado parâmetros adicionais na escolha do dispositivo? [S/N]') # Mensagem ao usuário questionando se houve a necessidade de inserção de parâmetros extras para a fonte de vídeo
	parametrocam = input('>') # Recebe a resposta do usuário sobre a fonte de vídeo

	# Se não foi necessário parâmetros adicionais, é possível informar mais parâmetros
	if parametrocam == "n" or parametrocam == "N":
		# Mensagem solicitando ao usuário a escolha e definição da resolução de captura e gravação do vídeo em tempo real
		print('\nÉ necessário definir a resolução (em pixels) para a captura. Digite o valor para o comprimento (width):')
		wcaptura = int(input('>')) # Armazena o valor da resolução de comprimento (width) em pixels fornecido pelo usuário
		vs.set(3, wcaptura) # Atribui na fonte de vídeo o valor digitado de resolução (em pixels) de comprimento (width)
		print('\nDigite o valor para a altura (height):')  # Mensagem solicitando ao usuário a escolha e definição da resolução (em pixels) de comprimento (height)
		hcaptura = int(input('>')) # Armazena o valor da resolução (em pixels) de altura (height) fornecido pelo usuário
		vs.set(4, hcaptura) # Atribui na fonte de vídeo o valor digitado de resolução (em pixels) de altura (height)

	# Mensagem solicitando ao usuário o local que será armazenado a captura do vídeo exibido na tela em tempo real
	print('\nDigite o nome (sem a extensão) e o local do arquivo que deseja armazenar o vídeo em tempo real original e com os pontos de rastreamento:')
	saidastream = input('>') # Recebe do usuário o local completo de onde será armazenado a captura do vídeo

else: # Opção digitada de maneira incorreta
	print('Opção inválida!') # Mensagem ao usuário dizendo que a opção digitada foi incorreta
	sys.exit() # Encerra o programa

# Opções referentes ao nome e local do arquivo escolhido para salvar os pontos de rastreamento
# Mensagem ao usuário solicitando o nome e o local do arquivo .csv a ser salvo
print('\nDigite o nome (sem a extensão) e o local do arquivo para salvar os dados de pontos de rastreamento:')
logarquivo = input('>') # Armazena o nome e o local que será salvo os dados de pontos de rastreamento

# Opções referentes à seleção da rede neural
print('\nDeseja-se usar a rede neural:') # Mensagem solicitando ao usuário a escolha da rede neural
print('1 - YOLOv3') # Usar a rede neural YOLOv3-tiny
print('2 - YOLOv3-tiny') # Usar a rede neural YOLOv3-tiny
print('3 - SSD Mobilenet-v2') # Usar a rede neural SSD Mobilenet-v2
opcaorede = int(input('>')) # Armazena a opção de rede neural escolhida pelo usuário

# Mensagens informativas sobre as estruturas das redes neurais
if opcaorede == 1 or opcaorede == 2: # Para a escolha da rede neural YOLOv3 e YOLOv3-tiny
	if opcaorede == 1:
		print('\nA estrutura da rede YOLOv3 deve ser à seguinte:')  # Mensagem demonstrando a estrutura exata que a pasta com a rede neural YOLOv3 deve conter
		print('|--- YOLOv3')  # Pasta que contêm os arquivos da rede neural YOLOv3
		print('|	|--- classes.names')  # Arquivo de listagem e nomes das classes
		print('|	|--- yolov3.weights')  # Arquivo de pesos
		print('|	|--- yolov3.cfg')  # Arquivo de configuração da rede neural YOLOv3
	if opcaorede == 2:
		print('\nA estrutura da rede YOLOv3-tiny deve ser à seguinte:') # Mensagem demonstrando a estrutura exata que a pasta com a rede neural YOLOv3-tiny deve conter
		print('|--- YOLOv3-tiny') # Pasta que contêm os arquivos da rede neural YOLOv3-tiny
		print('|	|--- classes.names') # Arquivo de listagem e nomes das classes
		print('|	|--- yolov3-tiny.weights') # Arquivo de pesos
		print('|	|--- yolov3-tiny.cfg') # Arquivo de configuração da rede neural YOLOv3-tiny

elif opcaorede == 3: # Para a escolha da rede neural SSD Mobilenet-v2
	print('\nA estrutura da rede SSD Mobilenet-v2 deve ser à seguinte:') # Mensagem demonstrando a estrutura exata que a pasta com a rede neural SSD Mobilenet-v2 deve conter
	print('|--- Mobilenetv2') # Pasta que contêm os arquivos da rede neural SSD Mobilenet-v2
	print('|	|--- classes.pbtxt') # Arquivo de listagem e nomes das classes
	print('|	|--- frozen_inference_graph.pb') # Arquivo de pesos

else: # Opção digitada de maneira incorreta
	print('\nOpção inválida!\n') # Mensagem ao usuário dizendo que a opção digitada foi incorreta
	sys.exit() # Encerra o programa

# Seleção da pasta de rede neural escolhida
print('\nIndique a pasta onde está localizado os arquivos da rede neural escolhida:') # Mensagem solicitando ao usuário indicar a pasta da rede neural escolhida
pastarede = input('>') # Armazena o local da rede neural escolhida

# Tratamento de localização dos arquivos necessários referentes à rede neural escolhida anteriormente
# Para o caso das redes neurais YOLOv3 e YOLOv3-tiny
if opcaorede == 1 or opcaorede == 2:
	if opcaorede == 1:
		# Verifica se o local da pasta informado contêm os arquivos da rede neural YOLOv3
		if not (os.path.isfile(pastarede + "/classes.names") and os.path.isfile(pastarede + "/yolov3.weights") and os.path.isfile(pastarede + "/yolov3.cfg")):
			# Mensagem informando ao usuário que os arquivos da rede neural YOLOv3 não foram encontrados ou não estão no formato estruturado informado
			print('\nArquivos da rede neural YOLOv3 não encontrados! Por favor, verifique se o local informado da pasta e a estrutura estão corretos.\n')
			sys.exit() # Encerra o programa
		print('\nArquivos da rede neural YOLOv3 encontrados!\n') # Mensagem informando ao usuário que os arquivos da rede neural YOLOv3 foram localizados

		# Carrega os arquivos de labels, pesos e configurações da rede YOLOv3
		rotulos = os.path.sep.join([pastarede, "classes.names"]) # Carrega o arquivo de nomes/classes
		rotuloslidos = open(rotulos).read().strip().split("\n") # Faz a leitura e varredura do arquivo de nomes/classes
		pesos = os.path.sep.join([pastarede, "yolov3.weights"]) # Carrega o arquivo de pesos
		configuracao = os.path.sep.join([pastarede, "yolov3.cfg"]) # Carrega o arquivo de configuração da rede

		# Etapa de carregamento da rede neural YOLOv3
		print('Carregando a rede YOLOv3...\n') # Mensagem informativa de carregamento da rede neural YOLOv3
		net = cv2.dnn.readNetFromDarknet(configuracao, pesos) # Carrega a rede neural YOLOv3 com os arquivos de configuração e pesos informados pelo usuário

	if opcaorede == 2:
		# Verifica se o local da pasta informado contêm os arquivos da rede neural YOLOv3-tiny
		if not (os.path.isfile(pastarede + "/classes.names") and os.path.isfile(pastarede + "/yolov3-tiny.weights") and os.path.isfile(pastarede + "/yolov3-tiny.cfg")):
			# Mensagem informando ao usuário que os arquivos da rede neural YOLOv3-tiny não foram encontrados ou não estão no formato estruturado informado
			print('\nArquivos da rede neural YOLOv3-tiny não encontrados! Por favor, verifique se o local informado da pasta e a estrutura estão corretos.\n')
			sys.exit()  # Encerra o programa
		print('\nArquivos da rede neural YOLOv3-tiny encontrados!\n')  # Mensagem informando ao usuário que os arquivos da rede neural YOLOv3-tiny foram localizados

		# Carrega os arquivos de labels, pesos e configurações da rede YOLOv3-tiny
		rotulos = os.path.sep.join([pastarede, "classes.names"])  # Carrega o arquivo de nomes/classes
		rotuloslidos = open(rotulos).read().strip().split("\n")  # Faz a leitura e varredura do arquivo de nomes/classes
		pesos = os.path.sep.join([pastarede, "yolov3-tiny.weights"])  # Carrega o arquivo de pesos
		configuracao = os.path.sep.join([pastarede, "yolov3-tiny.cfg"])  # Carrega o arquivo de configuração da rede

		# Etapa de carregamento da rede neural YOLOv3-tiny
		print('Carregando a rede YOLOv3-tiny...\n')  # Mensagem informativa de carregamento da rede neural YOLOv3-tiny
		net = cv2.dnn.readNetFromDarknet(configuracao, pesos)  # Carrega a rede neural YOLOv3-tiny com os arquivos de configuração e pesos informados pelo usuário

	# Verifica se a GPU deve ser usada ou não no processamento
	if escolhaproc == "GPU" or escolhaproc == "gpu": # Caso queira executar com a GPU
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # Atribui que será utilizado a GPU como backend para o processamento
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # Atribui o uso do CUDA para o processamento

	listacamadas = net.getLayerNames() # Obtêm os nomes de todas as camadas da rede neural YOLOv3-tiny
	listacamadas = [listacamadas[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # Obtêm os nomes das camadas YOLO de saída advindas da rede neural YOLOv3-tiny

elif opcaorede == 3: # Para o caso da rede SSD Mobilenet-v2
	# Verifica se o local da pasta informado contêm os arquivos da rede neural SSD Mobilenet-v2
	if not (os.path.isfile(pastarede + "/classes.pbtxt") and os.path.isfile(pastarede + "/frozen_inference_graph.pb")):
		# Mensagem informando ao usuário que os arquivos da rede neural SSD Mobilenet-v2 não foram encontrados ou não estão no formato estruturado informado
		print('\nArquivos da rede neural SSD Mobilenet-v2 não encontrados! Por favor, verifique se o local informado da pasta e a estrutura estão corretos.\n')
		sys.exit() # Encerra o programa
	print('\nArquivos da rede neural SSD Mobilenet-v2 encontrados!\n') # Mensagem informando ao usuário que os arquivos da rede neural SSD Mobilenet-v2 foram localizados

	# Carrega os arquivos de labels, pesos e configurações da rede SSD Mobilenet-v2
	rotulos = os.path.sep.join([pastarede, "classes.pbtxt"]) # Carrega o arquivo de nomes/classes
	numeroclasses = 90 # Número de classes para o modelo
	pesos = os.path.sep.join([pastarede, "frozen_inference_graph.pb"]) # Carrega o arquivo de topologia e pesos

	# Torna o modelo como principal/padrão para a execução
	with modelo.as_default():
		# Inicializa o gráfico de definições, utilizado para serializar o gráfico computacional do TensorFlow
		defgrafico = tf.GraphDef()

		# Carregamento do arquivo de topologia e pesos
		with tf.gfile.GFile(pesos, "rb") as f:
			graficoserial = f.read() # Realiza a leitura do arquivo de topologia e pesos
			defgrafico.ParseFromString(graficoserial) # Serializa o gráfico computacional
			tf.import_graph_def(defgrafico, name="") # Realiza a importação do gráfico computacional serializado

	# Carrega o arquivo de classes e rótulos a partir do disco
	maparotulos = label_map_util.load_labelmap(rotulos) # Carrega o arquivo de classes e rótulos
	categorias = label_map_util.convert_label_map_to_categories(maparotulos, numeroclasses, use_display_name=True) # Converte o mapa de rótulos e classes, retornando uma lista de dicionários
	categoriaid = label_map_util.create_category_index(categorias) # Cria um dicionário codificado pelo ID da categoria

else: # Para qualquer outro valor é retornado ao usuário que a opção é inválida
	print('\nOpção inválida!\n') # Mensagem ao usuário dizendo que a opção digitada foi incorreta
	sys.exit() # Encerra o programa

# Opções referentes à seleção do rastreador
print('Deseja-se usar o rastreador:') # Mensagem solicitando ao usuário a escolha do rastreador
print('1 - dlib') # Usar o rastreador dlib
print('2 - CSRT') # Usar o rastreador CSRT
print('3 - KCF') # Usar o rastreador KCF
print('4 - MOSSE') # Usar o rastreador MOSSE
opcaorastreador = input('>') # Armazena a opção de rastreador escolhida pelo usuário

if opcaorastreador != "1":
	# Inicializa um dicionário que liga o rastreador do OpenCV de acordo com a escolha do usuário
	OPENCV_OBJECT_TRACKERS = {
		"2": cv2.TrackerCSRT_create, #top1, um pouco de queda de FPS e alta acurácia
		"3": cv2.TrackerKCF_create, #top3, um pouco de queda de FPS, acurácia baixa (se perde baixo)
		"4": cv2.TrackerMOSSE_create #top2, muito rápido, acurácia alta
	}

	# Inicializa a função de rastreamento de múltiplos objetos do OpenCV
	multirastreadores = cv2.MultiTracker_create()

# Inicializa a lista de pontos de rastreamento
listapontos = deque()

# Inicializa o parâmetro de confiança, visando filtrar detecções fracas (abaixo de 30%)
filtroconfiancayolo = 0.3
filtroconfiancamn = 0.5

# Inicializa o parâmetro de limite (threshold) 
valorlimite = 0.5

# Inicializa uma variável auxiliar que permite a gravação do vídeo
gravarvideo = None
gravarvideorastreio = None

# Inicializa as variáveis de contagem de FPS
aquecimento = 0
contagemfps = 0 # Inicializa uma variável auxiliar para a contagem de FPS
totalFPS = 0 # Inicializa a variável para a contagem total de FPS
totalframes = 0 # Inicializa a variável para a quantidade de frames lidos e processados

# Inicializações do rastreamento
objetos = OrderedDict() # Inicializa e declara um dicionário de objetos
desaparecidos = OrderedDict() # Inicializa e declara um dicionário de objetos desaparecidos
valordesaparece = 40 # Inicializa o parâmetro que define em quantos frames o objeto será considerado perdido
valordistancia = 50 # Inicializa o parâmetro que define a distância entre objetos
saltoframe = 30 # Inicializa o parâmetro que define de quantos em quantos frames será realizada a detecção
IDobjetoaux = 0 # Inicializa a variável de vinculação da ID do objeto
porcentagens = [] # Inicializa a variável de armazenamento das porcentagens
rastreadores = [] # Inicializa a variável de armazenamento dos rastreadores
caixadetectada = []
flagrastreador = 0
rotulosrastreados = [] # Inicializa a variável de armazenamento dos rótulos
objetosrastreados = {} # Inicializa a variável de objetos rastreáveis

# create a session to perform inference
with modelo.as_default():
	with tf.Session(graph=modelo) as sessao:
		# Inicia o loop
		while True:
			# Inicia a contagem dos frames por segundo utilizando o estimador da biblioteca imutils
			contaframe = FPS().start()

			# Realiza a leitura do frame atual e do próximo frame, para que possa ser executado tanto para vídeo gravado quanto em tempo real
			frame = vs.read()  # Captura o frame atual do vídeo
			frame = frame[1]  # Captura o próximo frame e o armazena

			# Tratamento para verificação de final do vídeo
			if frame is None:
				break

			# Obtêm as dimensões do frame lido
			(altura, comprimento) = frame.shape[:2]

			if opcaofonte == 2:
				# Copia o frame de entrada afim de manter o frame de captura original, sem informações de rastreamento, utilizado apenas para gravações em tempo real
				frameoriginal = frame.copy()

			# Converte o frame atual para o espaço de cores RGB, para que o rastreador do dlib possa operar corretamente
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Inicializa a lista de armazenamento das caixas delimitadoras advindas do detector ou do rastreador
			retangulos = []

			# Tratamento para aplicação da detecção ou rastreamento dependendo da quantidade de frames percorridos
			# Utiliza-se o detector, para obter os objetos e suas localizações
			if totalframes % saltoframe == 0:
				# Inicializa os rastreadores
				if opcaorastreador != "1":
					multirastreadores = cv2.MultiTracker_create()

				# Inicializa a variável de armazenamento dos rastreadores
				rastreadores = []

				# Caso a rede escolhida seja a YOLOv3 ou YOLOv3-tiny
				if opcaorede == 1 or opcaorede == 2:
					# Processamento da rede
					preprocessamento = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)  # Realiza o pré-processamento do frame antes de ser passada adiante na rede neural
					net.setInput(preprocessamento)  # Utiliza o frame pré-processado como entrada da rede
					camadasaida = net.forward(listacamadas)  # Realiza uma passagem direta no detector de objetos YOLO, retornando a saída com as devidas probabilidades associadas as caixas delimitadoras

					# Inicializações das listas de caixas delimitadoras, confianças e lista de classses dos objetos
					caixasdelimitadoras = []  # Inicializa a lista das caixas delimitadoras (bounding boxes) detectadas
					listaconfiancas = []  # Inicializa a lista de confianças detectadas para cada um dos objetos
					classesids = []  # Inicializa a lista de rótulos (labels) dos objetos identificados

					# Realiza um laço de repetição (loop) em cada uma das camadas de saídas
					for saida in camadasaida:
						# Realiza um laço de repetição (loop) em cada uma detecções
						for deteccao in saida:
							# Realiza a extração das classes de identificação dos objetos em conjunto com as probabilidades do objeto detectado
							pontuacoes = deteccao[5:]  # Armazena as probabilidades de detecções
							classeid = np.argmax(pontuacoes)  # Extraí o índice que consta o objeto com a maior probabilidade dentre os quais foram detectados
							valorconfianca = pontuacoes[classeid]  # Salva a referida probabilidade do objeto detectado

							# Filtra as probabilidades fracas, verificando que estas são maiores que o mínimo delimitado pelo filtro
							if valorconfianca > filtroconfiancayolo:
								# Redimensiona as coordenadas da caixa delimitadora relativas ao tamanho da imagem
								caixa = deteccao[0:4] * np.array([comprimento, altura, comprimento, altura])  # Armazena as coordenadas da caixa delimitadora com os valores proporcionais a imagem
								(comecoX, comecoY, fimX, fimY) = caixa.astype("int")  # Define as coordenadas da caixa delimitadoras como int
								pontoscentro = (comecoX, comecoY)  # Armazena as coordenadas centrais de referência X e Y

								# Por meio das coordenadas centrais, obtêm o topo e o canto esquerdo da caixa delimitadora
								comecoX = int(comecoX - (fimX / 2))  # Obtêm a coordenada em X
								comecoY = int(comecoY - (fimY / 2))  # Obtêm a coordenada em Y

								# Atualiza a lista com as coordenadas das caixas delimitadoras, confianças e a lista de rótulos dos objetos identificados
								caixasdelimitadoras.append([comecoX, comecoY, int(fimX), int(fimY)])  # Inclui na lista as coordenadas da caixa
								listaconfiancas.append(float(valorconfianca))  # Inclui na lista as confianças obtidas na identificação
								classesids.append(classeid)  # Inclui na lista os rótulos dos objetos identificados

					# Aplica a função de supressão não-máxima (NMS), a fim de suprimir as caixas delimitadoras de baixa confiança e sobrepostas
					identificacoes = cv2.dnn.NMSBoxes(caixasdelimitadoras, listaconfiancas, filtroconfiancayolo, valorlimite)

					# Verifica se existe pelo menos uma detecção
					if len(identificacoes) > 0:
						# Realiza um laço de repetições (loop) sobre as detecções obtidas
						for i in identificacoes.flatten():
							# Extraí as coordenadas da caixa delimitadora
							(comecoX, comecoY) = (caixasdelimitadoras[i][0], caixasdelimitadoras[i][1])  # Extrai as coordenadas em X e Y a partir da lista de detecções
							(fimX, fimY) = (caixasdelimitadoras[i][2], caixasdelimitadoras[i][3])  # Extrai o tamanho da caixa delimitadora, comprimento (width) e altura (height), a partir da lista de detecções

							if opcaorastreador != "1":
								# create a new object tracker for the bounding box and add it to our multi-object tracker
								caixadetectada = (comecoX, comecoY, fimX, fimY)
								rastreador = OPENCV_OBJECT_TRACKERS[opcaorastreador]()
								multirastreadores.add(rastreador, frame, caixadetectada)
								porcentagens.append(listaconfiancas[i])  # Armazena as porcentagens atribuídas aos objetos detectados
								rotulosrastreados.append(rotuloslidos[classesids[i]])  # Armazena os rótulos atribuídos aos objetos detectados
							else:
								# Operações do rastreador
								rastreador = dlib.correlation_tracker()  # Inicializa o rastreador de correlação do dlib
								retangulo = dlib.rectangle(comecoX, comecoY, comecoX + fimX, comecoY + fimY)  # Constrói o objeto retângular do dlib a partir das coordenadas da caixa delimitadora
								rastreador.start_track(rgb, retangulo)  # Inicia o processo de rastreamento
								rastreadores.append(rastreador)  # Armazena os rastreadores atuais na lista de rastreadores, para que possa ser utilizado durante o pulo de frame
								porcentagens.append(listaconfiancas[i])  # Armazena as porcentagens atribuídas aos objetos detectados
								rotulosrastreados.append(rotuloslidos[classesids[i]])  # Armazena os rótulos atribuídos aos objetos detectados

				# Caso a rede escolhida seja a SSD Mobilenet-v2
				if opcaorede == 3:
					imagemtensor = modelo.get_tensor_by_name("image_tensor:0")  # Obtêm a referência da imagem (frame) de entrada
					caixastensor = modelo.get_tensor_by_name("detection_boxes:0")  # Obtêm informações das caixas delimitadoras
					porcentagenstensor = modelo.get_tensor_by_name("detection_scores:0")  # Obtêm informações de pontuações (porcentagens) da detecção
					classestensor = modelo.get_tensor_by_name("detection_classes:0")  # Obtêm informações das classes
					numerodeteccoes = modelo.get_tensor_by_name("num_detections:0")  # Obtêm o número de detecções

					# Manipulações da imagem (frame) de entrada
					frameaux = np.expand_dims(rgb, axis=0)  # Expande as dimensões do vetor da imagem

					# Realiza a inferência/detecção, computando as caixas delimiitadoras, probabilidades e os rótulos das classes
					(caixas, pontuacoes, labels, N) = sessao.run([caixastensor, porcentagenstensor, classestensor, numerodeteccoes], feed_dict={imagemtensor: frameaux})

					# Reduz as listas em uma única dimensão
					caixas = np.squeeze(caixas)  # Reduz a dimensão da lista de caixas delimitadoras
					pontuacoes = np.squeeze(pontuacoes)  # Reduz a dimensão da lista das pontuações (porcentagens)
					labels = np.squeeze(labels)  # Reduz a dimensão da lista de rótulos (labels)

					# Faz um laço de repetição (loop) sobre as predições das caixas delimitadoras obtidas
					for (caixa, pontuacao, label) in zip(caixas, pontuacoes, labels):
						# Se a pontuação (probabilidade) for menor que a confiança mínima estipulada, ignorá-la
						if pontuacao < filtroconfiancamn:
							continue

						# Desenha a predição, porcentagem e a caixa delimitadora na imagem de saída
						label = categoriaid[label]  # Armazena a identificação e o nome da classe
						identificacao = int(label["id"])  # Armazena apenas o número de identificação da classe

						# Redimensiona a caixa delimitadora para a faixa entre 0 à 1 para o comprimento (width) e altura (height) e calcula os pontos centrais
						(comecoY, comecoX, fimY, fimX) = caixa  # Extrai as coordenadas da caixa delimitadora
						comecoX = int(comecoX * comprimento)  # Obtêm a coordenada inicial (em pixels) em X
						comecoY = int(comecoY * altura)  # Obtêm a coordenada inicial (em pixels) em Y
						fimX = int(fimX * comprimento)  # Obtêm a coordenada final (em pixels) em X
						fimY = int(fimY * altura)  # Obtêm a coordenada final (em pixels) em Y
						fimX = int(fimX - comecoX) # Correção da coordenada fimX
						fimY = int(fimY - comecoY) # Correção da coordenada fimY

						if opcaorastreador != "1":
							caixadetectada = (comecoX, comecoY, fimX, fimY)
							rastreador = OPENCV_OBJECT_TRACKERS[opcaorastreador]()
							multirastreadores.add(rastreador, frame, caixadetectada) # create a new object tracker for the bounding box and add it to our multi-object tracker
							porcentagens.append(pontuacao)  # Armazena as porcentagens atribuídas aos objetos detectados
							rotulosrastreados.append(label["name"])  # Armazena os rótulos atribuídos aos objetos detectados
						else:
							# Operações do rastreador
							rastreador = dlib.correlation_tracker()  # Inicializa o rastreador de correlação do dlib
							retangulo = dlib.rectangle(comecoX, comecoY, comecoX + fimX, comecoY + fimY)  # Constrói o objeto retângular do dlib a partir das coordenadas da caixa delimitadora
							rastreador.start_track(rgb, retangulo)  # Inicia o processo de rastreamento
							rastreadores.append(rastreador)  # Armazena os rastreadores atuais na lista de rastreadores, para que possa ser utilizado durante o pulo de frame
							porcentagens.append(pontuacao)  # Armazena as porcentagens atribuídas aos objetos detectados
							rotulosrastreados.append(label["name"])  # Armazena os rótulos atribuídos aos objetos detectados

			# Utiliza-se o rastreador ao invés do detector, garantindo maior desempenho
			else:
				if opcaorastreador != "1":
					(success, rastreadores) = multirastreadores.update(frame)
				# Percorre a lista de rastreadores, rótulos e porcentagens
				for (rastreador, rotulo, porcentagem) in zip(rastreadores, rotulosrastreados, porcentagens):
					if opcaorastreador == "1":
						# Atualizações do rastreador
						rastreador.update(rgb)  # Atualiza o rastreador
						pos = rastreador.get_position()  # Obtêm a posição atualizada do objeto

						# Obtêm as coordenadas do objeto em processo de rastreamento
						comecoX = int(pos.left())  # Armazena a posição de início da caixa em X (Esquerda)
						comecoY = int(pos.top())  # Armazena a posição de início da caixa em Y (Topo)
						fimX = int(pos.right())  # Armazena a posição de fim da caixa em X (Direita)
						fimY = int(pos.bottom())  # Armazena a posição de fim da caixa em Y (Inferior)

						# Armazena as coordenadas obtidas anteriormente na lista de retângulos
						retangulos.append((comecoX, comecoY, fimX, fimY))
					else:
						# Obtêm as coordenadas do objeto em processo de rastreamento
						(x, y, w, h) = [int(v) for v in rastreador]
						comecoX = int(x)  # Armazena a posição de início da caixa em X (Esquerda)
						comecoY = int(y)  # Armazena a posição de início da caixa em Y (Topo)
						fimX = int(x + w)  # Armazena a posição de fim da caixa em X (Direita)
						fimY = int(y + h)  # Armazena a posição de fim da caixa em Y (Inferior)

						# Armazena as coordenadas obtidas anteriormente na lista de retângulos
						retangulos.append((comecoX, comecoY, fimX, fimY))

					# Exibição da caixa delimitadora, porcentagens e rótulo dos objetos rastreados
					rotuloimprime = "{}: {:.2f}".format(rotulo, porcentagem)  # Armazena a classe referente aos objetos identificados e suas probabilidades (confiança)
					cv2.rectangle(frame, (comecoX, comecoY), (fimX, fimY), (0, 0, 255), 2)  # Exibe os retângulos de determinada cor para a classe
					cv2.putText(frame, rotuloimprime, (comecoX, comecoY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)  # Exibe a classe e as porcentagens para o usuário

			# Tratamento de verificação da lista de retângulos de caixas delimitadoras
			# Verifica se a lista está vazia
			if len(retangulos) == 0:
				# Percorre sobre os objetos rastreados existentes, sinalizando-os como desaparecidos
				for IDobjeto in list(desaparecidos.keys()):
					desaparecidos[IDobjeto] += 1  # Realiza a contagem para o objeto perdido

					# Caso foi chegado à um número máximo de frames consecutivos onde um objeto foi sinalizado como perdido, apaga-se o objeto da lista
					if desaparecidos[IDobjeto] > valordesaparece:
						del objetos[IDobjeto]  # Apaga o objeto da lista
						del desaparecidos[IDobjeto]  # Apaga a lista de desaparecidos

			# Para o caso de a lista não estar vazia
			else:
				# Inicializa uma lista de centroides para o frame atual
				centroideatual = np.zeros((len(retangulos), 2), dtype="int")

				# Percorre sobre os retângulos de caixas delimitadoras
				for (i, (comecoX, comecoY, fimX, fimY)) in enumerate(retangulos):
					# Obtêm os pontos da centroide a partir das coordenadas das caixas delimitadoras
					centroX = int((comecoX + fimX) / 2.0)  # Obtêm a coordenada central em X
					centroY = int((comecoY + fimY) / 2.0)  # Obtêm a coordenada central em Y
					centroideatual[i] = (centroX, centroY)  # Armazena as coordenadas das centroides em uma lista

				# Se não está sendo rastreado quaisquer objetos, utiliza-se a centroide de entrada e registra-se cada uma
				if len(objetos) == 0:
					for i in range(0, len(centroideatual)):
						centroide = centroideatual[i]  # Armazena a centroide calculada
						objetos[IDobjetoaux] = centroide  # Vincula a centroide com o objeto
						desaparecidos[IDobjetoaux] = 0  # Reseta o contador de desaparecimento
						IDobjetoaux += 1  # Acrescenta a variável auxiliar de vinculação de objetos

				# Caso contrário, estamos rastreando os objetos, assim, precisamos conferir as centroides de entrada com as centroides existentes
				else:
					# Obtêm a ID deos objetos e suas centroides
					IDsobjetos = list(objetos.keys())  # Obtêm a ID dos objetos
					centroidesobjetos = list(objetos.values())  # Obtêm as centroides dos objetos

					# Calcula a distância entre as centroides de entrada e as centroides atuais, tentando verificar se as mesmas estão próximas e se conferem aos devidos objetos
					distcentroide = dist.cdist(np.array(centroidesobjetos), centroideatual)

					# Encontra-se o menor valor da linha e organiza os índices baseados em seus valores mínimos, assim, o menor valor se encontrará de início no índice da lista
					linhas = distcentroide.min(axis=1).argsort()

					# Encontra-se o menor valor da coluna e organiza os índices baseados nos índices das linhas
					colunas = distcentroide.argmin(axis=1)[linhas]

					# Para verificar se um objeto precisa ser atualizado, registrado ou apagado, é necessário verificar quais colunas e linhas foram analisadas
					linhausada = set()  # Verifica as linhas
					colunausada = set()  # Verifica as colunas

					# Percorre sobre a combinação (linhas, colunas) de índices de tuplas
					for (linha, coluna) in zip(linhas, colunas):
						# Se as linhas ou colunas foram analisadas anteriormente, ignorá-las
						if linha in linhausada or coluna in colunausada:
							continue

						# Se a distância entre centroides é maior que a distância máxima, não associar as duas centroides com o mesmo objeto
						if distcentroide[linha, coluna] > valordistancia:
							continue

						# Caso contrário, obter a ID do objeto com sua linha associada, definindo sua nova centroide, e resetar o contador de desaparecimento
						IDobjeto = IDsobjetos[linha]  # Obtem a ID do objeto de acordo com sua linha
						objetos[IDobjeto] = centroideatual[coluna]  # Vincula a nova centroide
						desaparecidos[IDobjeto] = 0  # Reseta o contador de desaparecimento

						# Indica que cada índice de linha e coluna foi analisado
						linhausada.add(linha)  # Indica a linha
						colunausada.add(coluna)  # Indica a coluna

					# Calcula ambos os índices de linha e coluna que ainda não foram examidados
					linhasnulas = set(range(0, distcentroide.shape[0])).difference(linhausada)  # Calcula a linha
					colunasnulas = set(range(0, distcentroide.shape[1])).difference(colunausada)  # Calcula a coluna

					# Tratamento para o caso do número de centroides for maior ou igual ao número de centroides de entrada, verificando se os objetos podem ter desaparecido
					if distcentroide.shape[0] >= distcentroide.shape[1]:
						# Percorre sobre os índices de linhas que não foram utilizados
						for linha in linhasnulas:
							# Obtêm a ID do objeto e seu índice de linha correspondente e incrementa o contador de desaparecimento
							IDobjeto = IDsobjetos[linha]  # Obtêm a ID do objeto
							desaparecidos[
								IDobjeto] += 1  # Realiza a contagem de frames consecutivos relacionados ao desaparecimento

							# Verifica se o número de frames consecutivos para o objeto foi sinalizado como desaparecido e, por garantia, apaga o objeto
							if desaparecidos[IDobjeto] > valordesaparece:
								del objetos[IDobjeto]  # Deleta o objeto
								del desaparecidos[IDobjeto]  # Deleta da lista de desaparecido

					# Caso contrário, o número de centroides de entrada é maior que o número de centroides dos objetos já existentes, portanto, é preciso registrar as centroides de entrada como objeto rastrável
					else:
						# Percorre sobre os índices de colunas que não foram utilizados
						for coluna in colunasnulas:
							centroide = centroideatual[coluna]  # Armazena a centroide
							objetos[IDobjetoaux] = centroide  # Vincula a centroide com o objeto
							desaparecidos[IDobjetoaux] = 0  # Reseta o contador de desaparecimento
							IDobjetoaux += 1  # Acrescenta a variável auxiliar de vinculação de objetos

			# Percorre sobre os objetos rastreados e suas centroides
			for (IDobjeto, centroide) in objetos.items():
				# Verifica se o objeto rastreável existe para a ID atual
				to = objetosrastreados.get(IDobjeto, None)

				# Caso não há um objeto rastreável, crie-o
				if to is None:
					IDobjeto = IDobjeto  # Vincule com a ID do objeto
					centroids = [centroide]  # Passe as coordenadas da centroide
				# Caso exista, armazene sua centroide
				else:
					centroids.append(centroide)  # Armazena as coordenadas da centroide

				# Armazena o objeto rastreável no dicionário
				objetosrastreados[IDobjeto] = to

				# Manipulações de desenho da ID e o ponto central do objeto
				txt = "ID {}".format(IDobjeto)  # Armazena a ID do objeto em formato de texto para imprimir na tela ao usuário
				cv2.putText(frame, txt, (centroide[0] - 10, centroide[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Escreve o número da ID do objeto
				cv2.circle(frame, (centroide[0], centroide[1]), 4, (0, 0, 255), -1)  # Desenha um pequeno círculo que indica o centro da caixa delimitadora

				# Armazena a ID do objeto e as coordenadas do ponto central
				listapontos.append((IDobjeto, centroide[0], centroide[1]))

			# Tratamento para a geração do arquivo de vídeo contendo as informações de rastreamento a partir de um arquivo de vídeo previamente gravado
			if gravarvideorastreio is None and opcaofonte == 1:
				formatovideo = cv2.VideoWriter_fourcc(*'XVID')  # Define o codec para gravação/compactação do arquivo de vídeo em tempo real
				# Para testar outros codecs, consulte a página: http://www.fourcc.org/codecs.php
				gravarvideorastreio = cv2.VideoWriter(saidastream + "_rastreio.avi", formatovideo, fpscerto, (comprimento, altura), True)  # Habilita a gravação do arquivo de vídeo

			# Tratamento para a gravação do arquivo de vídeo em tempo real (vídeo RAW e informações de rastreamento)
			if gravarvideo is None and gravarvideorastreio is None and opcaofonte == 2 and aquecimento >= 11:  # Caso o arquivo não esteja presente, sem gravações, for escolhido a opção em tempo real e a aplicação estabilizada
				formatovideo = cv2.VideoWriter_fourcc(*'XVID')  # Define o codec para gravação/compactação do arquivo de vídeo em tempo real
				# Para testar outros codecs, consulte a página: http://www.fourcc.org/codecs.php
				gravarvideo = cv2.VideoWriter(saidastream + "_original.avi", formatovideo, fpscerto, (comprimento, altura), True)  # Habilita a gravação do arquivo de vídeo
				gravarvideorastreio = cv2.VideoWriter(saidastream + "_rastreio.avi", formatovideo, fpscerto, (comprimento, altura), True)  # Habilita a gravação do arquivo de vídeo

			# Verifica a quantidade de vezs que foi passado pelo loop para ser iniciado a gravação no arquivo com o FPS já estabelecido
			if opcaofonte == 1:  # Caso seja escolhido a opção em tempo real e a aplicação tenha estabilizado
				gravarvideorastreio.write(frame)  # Escreve o arquivo de vídeo com os dados presentes no frame atual (frame do vídeo em tempo real) sem os dados de rastreamento (caixas delimitadoras)

			# Verifica a quantidade de vezs que foi passado pelo loop para ser iniciado a gravação no arquivo com o FPS já estabelecido
			if opcaofonte == 2 and aquecimento >= 11:  # Caso seja escolhido a opção em tempo real e a aplicação tenha estabilizado
				gravarvideo.write(frameoriginal)  # Escreve o arquivo de vídeo com os dados presentes no frame atual (frame do vídeo em tempo real) com os dados de rastreamento (caixas delimitadoras)
				gravarvideorastreio.write(frame)  # Escreve o arquivo de vídeo com os dados presentes no frame atual (frame do vídeo em tempo real) sem os dados de rastreamento (caixas delimitadoras)

			# Exibe a fonte de vídeo selecionada e a janela de rastreamento ao usuário
			cv2.imshow("Rastreamento de Peixes", frame)  # Exibe a fonte de vídeo selecionada

			# Fica observando caso seja pressionado alguma tecla do teclado
			teclado = cv2.waitKey(30) & 0xFF

			# Se a tecla "q" for pressionada, o programa é encerrado
			if teclado == ord("q"):  # Caso a tecla "q" seja pressionada
				break  # Encerra o loop

			# Manipulações de operações de frames
			contaframe.update()  # Atualiza a contagem de FPS
			totalframes += 1  # Realiza a contagem de frames processados
			contaframe.stop()  # Para a atualização de FPS
			fpsatual = format(contaframe.fps())  # Obtêm o valor de FPS para o frame processado
			totalFPS = totalFPS + float(fpsatual)  # Realiza a soma total de FPS

			if opcaofonte == 2:
				# Tratamento para sincronização e obtenção do frame correto de gravação do vídeo em tempo real
				if 0 <= aquecimento <= 10:
					aquecimento = aquecimento + 1;  # Contador auxiliar de estabilização da gravação do vídeo
				if 6 <= aquecimento <= 10:  # Passado 5 contagens, o processamento do sistema encontra-se estabilizado
					contagemfps = contagemfps + float(fpsatual)  # Armazena 5 valores considerados estáveis de FPS
				if aquecimento >= 11:  # Depois de 11 contagens, obtêm-se o valor correto de FPS
					fpscerto = contagemfps / 5  # Realiza a média de 5 frames estáveis para encontrar o FPS certo de gravação do arquivo de vídeo
					fpscerto = round(fpscerto, 2)  # Obtêm a taxa de frames correta para gravação do vídeo

# Exibe a média de frames de execução do sistema
totalFPS = totalFPS / totalframes  # Realiza a média aritmética de FPS de execução da detecção e rastramento
totalFPS = round(totalFPS, 2)  # Arredonda o valor de FPS, fixando num valor de apenas 2 casas após a vírgula
print("FPS médio:", totalFPS)  # Mensagem informando ao usuário a média de FPS

# Grava os pontos de rastreamento para um arquivo
print('\nGravando o arquivo de pontos de rastreamento (.csv)...')  # Mensagem informando ao usuário que os pontos de rastreamento estão sendo gravados no arquivo .csv
pd.DataFrame(listapontos).to_csv(logarquivo + ".csv", index=False, header=['ID', 'X', 'Y'])  # Grava os pontos (coordenadas) de rastreamento no arquivo .csv definido pelo usuário
print('Pontos gravados!')  # Mensagem informando ao usuário que os pontos de rastreamento foram gravados no arquivo .csv
print('Encerrando a aplicação...')  # Mensagem informando ao usuário que a aplicação está em processo de encerramento

# Caso a fonte de vídeo seja em arquivo, para a leitura
if opcaofonte == 1:
	print('Fechando o arquivo de vídeo...')  # Mensagem informando ao usuário que o arquivo de vídeo está sendo fechado
	gravarvideorastreio.release()  # Fecha a conexão de gravação do arquivo de vídeo
	vs.release()  # Fecha a conexão com o arquivo de vídeo aberto

# Caso a fonte de vídeo seja em tempo real, encerra a transmissão e a gravação
else:
	print('Fechando a comunicação em tempo real...')  # Mensagem informando ao usuário que a fonte de vídeo em tempo real está sendo fechada
	gravarvideo.release()  # Fecha a conexão de gravação do arquivo de vídeo
	gravarvideorastreio.release()  # Fecha a conexão de gravação do arquivo de vídeo
	vs.release()  # Fecha a conexão com a fonte de vídeo em tempo real, liberando-a

# Fecha as janelas de vídeo e rastreamento
print('Programa finalizado!\n')  # Mensagem ao usuários informando que o programa foi encerrado
cv2.destroyAllWindows()  # Fecha todas as abertas janelas do OpenCV