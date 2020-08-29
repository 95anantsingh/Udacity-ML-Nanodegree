from threading import Thread as T
from threading import current_thread as ct
import time

a = True


def abc():
    b=0
    while a:
        print(b)
        time.sleep(0.5)
        b=b+1
    print('Stopping')


def stopper():
    global a
    time.sleep(1)
    a = False
    time.sleep(1)
    th.start()

def starter(start):
    while True:
        for i in range(0,1):
         t=T(target=abc)


if __name__ == '__main__':
    th = T(target=abd)
    st = T(target=stopper)
    st.start()
    th.start()
    th.join()
    print('stopped  ')