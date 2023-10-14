import time

def timing(func):
    def wrapper(*args, **kwargs):
        tic = time.perf_counter()
        res = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f'Total time to solve: {toc - tic} seconds.')
        return res, toc - tic
    return wrapper