import time
import threading 

def sleeper(i):
    print ("%d thread %d start time\n" % (i, time.time()))
    time.sleep(5)
    print ("%d thread %d stop time\n" % (i, time.time()))

for i in range(10):
    t = threading.Thread()
    lock = threading.Lock()
    lock.acquire()
    sleeper(i)
    lock.release()
    t.start()
    
    #t.join()