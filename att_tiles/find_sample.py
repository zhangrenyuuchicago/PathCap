import glob

for path in glob.glob('result/*.txt'):
    fin = open(path, 'r')
    for line in fin.readlines():
        line = line.strip()
        if line[0:2] == '[[':
            line = line[2:-2]
            array = line.split(',')
            for item in array:
                if abs(float(item) - 0.2) > 0.05:
                    print(path)
    fin.close()

