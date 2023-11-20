import numpy as np

def Valid(i,limit,flag,value) :
	if i>=0 and i<=limit-1 and (value[i] == 0 or value[i] == 2) and flag[i] == 0 :
		return True 
	return False	

def Checking(graph, start, end) :
	stack = []
	x,y = graph.shape
	graph1 = np.reshape(graph,-1)
	limit = len(graph1)
	flag = np.zeros([len(graph1)])

	stack.append(start[0]*x+start[1])
	flag[start[0]*x+start[1]] = 1
	while (len(stack)>0 and not flag[end[0]*x+end[1]] == 1) :
		element = stack.pop()
		#print element
		
		if Valid(element-1,limit,flag,graph1) and not element%x == 0 :
			stack.append(element-1)
			flag[element-1] = 1
		
		if Valid(element+1,limit,flag,graph1) and not (element+1)%x == 0:
			stack.append(element+1)
			flag[element+1] = 1
		
		if Valid(element-x,limit,flag,graph1) :
			stack.append(element-x)
			flag[element-x] = 1
		
		if Valid(element+x,limit,flag,graph1) :
			stack.append(element+x)
			flag[element+x] = 1

	# print(start, end)
	# print(start[0]*x+start[1])
	# print(end[0]*x+end[1])

	if flag[end[0]*x+end[1]] == 1 :
		return True
	return False	