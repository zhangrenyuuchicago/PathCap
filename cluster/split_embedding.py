import sys

fin = open('../AE_triplet_loss/gen_rep_II/representation_all_3epoch.txt', 'r')

pid_flag = {}
cur_pid = None
fout = None

while True:
    line = fin.readline().strip()
    if not line:
        break
    array = line.split()
    img_name = array[0]
    pid = img_name.split('_')[0]
    if pid not in pid_flag:
        fout = open('split/' + pid+'.txt', 'a')
        fout.write(line + '\n')
        fout.close()
    else:
        fout = open('split/' + pid+'.txt', 'w')
        fout.write(line + '\n')
        fout.close()

