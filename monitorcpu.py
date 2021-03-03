import argparse
import psutil

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--registro", type=str, help="Passa o nome do arquivo que será gerado os logs")
ap.add_argument("-i", "--idprocesso", type=int, help="Passa a ID do processo a ser monitorado")
args = vars(ap.parse_args())

p = psutil.Process(args["idprocesso"])

with open(args["registro"] + "CPULoad.csv", "a") as myfile:
    myfile.write("CPU,Memória\n")

    try:
        while True:
            myfile.write(str(p.cpu_percent(interval=0.033334)) + "%," + str(p.memory_percent()) + "%\n")

    except (AttributeError,FileNotFoundError,psutil.NoSuchProcess):
        print("\nO PID {} foi finalizado. Encerrando o registro de informações...\n".format(args["idprocesso"]))
        exit()
