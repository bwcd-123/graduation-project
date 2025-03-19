import os
import time
import swanlab
n = 1  # 设置重复次数
os.system("conda activate torch")


swanlab.login(api_key="1REF3sznzagTLaWp98nEZ")
for i in range(n):
    print("current iteration: ", i)
    start = time.time()
    os.system("python DAEGC/daegc.py --name Cora")
    os.system("python DAEGC/daegc.py --name Citeseer")
    end = time.time()
    print(f"current iteration: {i}time cost: {end-start:.4f}")