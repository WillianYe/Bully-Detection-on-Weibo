import json
import io
import sys


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
    with open("reviews_full.json",'r') as load_file :
        data = json.load(load_file)
    for i in range(5001):
        print(data[i])
