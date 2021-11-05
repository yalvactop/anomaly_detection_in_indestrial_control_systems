from multiprocessing import Queue
import multiprocessing
import time


def worker(q):
    while not q.empty():
        try:
            row = q.get(False)
            x, y = row
            print (fit(x, y))

        except Exception as e:
            print(e)
            break


def fit(x, y):#the fit()
    time.sleep(2)
    return str(x) + " x " + str(y) + " = " + str(x*y)

def main():
    print ('creating queue')
    q = multiprocessing.Queue()

    print ('enqueuing')
    for i in range(10):#nested loops of grid search are going to be here
        for j in range(10):
            q.put((i, j))

    num_processes = 5
    pool = []

    for i in range(num_processes):
        print ('launching process {0}'.format(i))
        p = multiprocessing.Process(target=worker, args=(q,))
        p.start()
        pool.append(p)

    for p in pool:
        p.join()

if __name__ == '__main__':
    main()