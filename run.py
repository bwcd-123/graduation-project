import os
import time
n = 100  # 设置重复次数
os.system("conda activate torch")
import swanlab

swanlab.login(api_key="1REF3sznzagTLaWp98nEZ")
for i in range(n):
    print("current iteration: ", i)
    start = time.time()
    os.system("python graphmae-so/train.py --dataset Cora --use_log")
    end = time.time()
    print(f"current iteration: {i}, time cost: {int(end-start)}s")