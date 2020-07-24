tuple_list = []
fin = open('sorted_word_freq.txt', 'r')
while True:
    line = fin.readline().strip()
    if not line:
        break
    line = line[1:-1]
    array = line.split(', ')
    word = array[0]
    freq = int(array[1])
    tuple_list.append((word, freq))

fin.close()

token_num = 0
total_num = 0
total_ge6_num = 0

for i in range(len(tuple_list)):
    tp = tuple_list[i]
    total_num += tp[1]
    if tp[1] > 5:
        token_num += 1
        total_ge6_num += tp[1]

print(total_ge6_num/ total_num)
print(f'token_num: {token_num}')
