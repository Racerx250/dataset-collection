import numpy as np

table = [[0 for i in range(11)] for j in range(11)]
table[0][0] = "size\\accuracy"
for i in range(1, 11):
	table[0][i] = i / 10
for i in range(1, 11):
	table[i][0] = (i - 1) / 10

with open('loopTable.txt') as openfileobject:
    for line in openfileobject:
    	if line[0] == 's':
        	size, accuracy = [int(i) for i in line.split() if i.isdigit()]
    	if line[5] == 'a':
        	accuracy_tmp = line.split()[2]
        	print(accuracy_tmp)
        	print(size, accuracy)
        	table[size + 1][accuracy] = accuracy_tmp


np.savetxt("table.csv", 
           table,
           delimiter =", ", 
           fmt ='% s')
