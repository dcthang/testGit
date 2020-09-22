fin = open("append-files_resize_3.txt", "r")
fout = open("append-files_resize_3_fixed.txt", "w")

for line in fin:
            line = line.split('\t')
            if len(line) == 3:
                del line[0]
                fout.write(line[0] + '\t' + line[1])
            else:
                fout.write('\n')
fout.close()
fin.close()
