import multiprocessing as mp
import numpy as np

def fun(x):
	print(f"Hello, this is {x} from process {mp.current_process()}")

list_ep = np.arange(10)
pool = mp.Pool(5)
pool.map(fun, list_ep)

