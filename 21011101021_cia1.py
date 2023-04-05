import random
char_set={'A','A','A','A','C','C','C','C','G','G','G','G','T','T','T','T'}
for i in range(0,16):
	s1=random.choice(char_set)
	s2=random.choice(char_Set)
mtch=5
mismtch=-4
matrix=[]
'''for i in range(0,16):
	if s1[i]==s2[i]:
		matrix[i][i]=match+matrix[i-3]
	elif s1[i]!=s2[i]:
		matrix[i][i]=
'''
def _match(s1,s2,matrix):
	for i in range(0,16):
		if s1[i]==s2[i]:
			matrix[i][i]=mtch+matrix[i-3][i-1]
			return mtch
		else:
			return 0
			
def _mismatch(s1,s2,matrix):
	for i in range(0,16):
		if s1[i]!=s2[i]:
			value1=matrix[i-3][i-1]-4
			value2=matrix[i-2][i-1]-4
			value3=matrix[i-3][i]-4
			if value1>value2 and value1>value3:
				matrix[i][i]=value1
			
			if value2>value1 and value2>value3:
				matrix[i][i]=value2
			
			if value3>value2 and value3>value1:
				matrix[i][i]=value3
				
			return mismtch
			
def backtrace(s1,s2,matrix):
    i = 16
    j = 16
    while (i>0 or j>0):
        if (i>0 and j>0 and matrix[i][j] == matrix[i-1][j-1] + _match(s1, s2, i, j)):
            sequ1 += s1[j-1]
            sequ2 += s2[i-1]
            i -= 1; j -= 1
        elif (i>0 and matrix[i][j] == Y[i][j]):
            sequ1 += '_'
            sequ2 += t[i-1]
            i -= 1
        elif (j>0 and matrix[i][j] == X[i][j]):
            sequ1 += s[j-1]
            sequ2 += '_'
			
def main():
	
