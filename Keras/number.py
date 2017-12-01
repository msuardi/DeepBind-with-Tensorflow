from os import listdir
import sys

if __name__ == '__main__':
    folder='perf/' + sys.argv[1]
    lista=listdir(folder)
    print(len(lista)-1)
