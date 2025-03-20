import os
import multiprocessing as mp
n = 100  # 设置重复次数
os.system("conda activate torch")
import swanlab

swanlab.login(api_key="1REF3sznzagTLaWp98nEZ")

def task():
    for i in range(n):
        os.system("python graphmae-so/train.py --dataset Cora --use_log")

if __name__ == "__main__":
    # 创建 4 个进程
    processes = []
    for i in range(4):
        p = mp.Process(target=task)
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("All processes completed!")