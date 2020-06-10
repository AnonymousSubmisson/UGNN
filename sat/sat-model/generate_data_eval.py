
from generator import generate_data
import multiprocessing
import os
import random
import pickle
import gzip
import time

q = multiprocessing.Queue()

class generate_data_config:
    SAVE_PATH = '/home/zhangwj/data/satevaldata_newgcn'
    NPROCESS = 8
    num_total = 10000
    num_each_min=10
    num_each_max=20
    rate_clauses_min=2.0
    rate_clauses_max=6.0
    rate_three=0.8

def generate_random_int():
    return random.randint(0, 2147483647)

def gendata_process(seed):
    while True:
        if q.empty():
            break
        i = q.get()
        filename = f'{i:08x}.gz'
        fullname = os.path.join(generate_data_config.SAVE_PATH, filename)
        try:
            gen = generate_data(i, generate_data_config.num_total, generate_data_config.num_each_min, generate_data_config.num_each_max, generate_data_config.rate_clauses_min, generate_data_config.rate_clauses_max, generate_data_config.rate_three)
            foutput = gzip.open(fullname, 'wb')
            pickle.dump(gen, foutput)
            foutput.close()
            print(filename)
        except:
            try:
                os.remove(fullname)
            except:
                pass
            raise

def main():
    os.makedirs(generate_data_config.SAVE_PATH, exist_ok=True)
    random.seed()

    processes = []

    for i in range(200):
        q.put(i)

    for _ in range(generate_data_config.NPROCESS):
        p = multiprocessing.Process(target=gendata_process, args=(generate_random_int(),))
        p.start()
        processes.append(p)

    while True:
        time.sleep(9999)

if __name__ == "__main__":
    main()
