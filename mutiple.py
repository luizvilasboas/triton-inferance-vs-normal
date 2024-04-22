import multiprocessing
import subprocess
import time
from utils import calculate_time

def run_script(script_name):
    subprocess.call('python ' + script_name, shell=True)

if __name__ == "__main__":
    script_name = "main.py --video videos/video-5-minutos.mp4 --triton"

    start_time = time.time()

    processes = []
    nums = 10
    for _ in range(nums):
        proc = multiprocessing.Process(target=run_script, args=(script_name,))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    end_time = time.time()

    total_seconds = end_time - start_time
    hours, minutes, seconds = calculate_time(total_seconds)

    print("> Todos rodaram")
    print(f"> O tempo total de Inferência de {nums} processos usando da forma normal foi de {hours} horas, {minutes} minutos, {seconds} segundos.")
