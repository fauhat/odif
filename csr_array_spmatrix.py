import numpy as np
from scipy.sparse import csr_array
from scipy.sparse import csr_matrix
from scipy.sparse import spmatrix

# csr_array_object:spmatrix = csr_array((3, 4), dtype=np.int8).toarray()
# csr_array_object:spmatrix = csr_matrix((data, indices, indptr),shape=(n_samples, self.node_count))
# print("csr_array_object: ", csr_array_object)

row = []
column = []
data = []
k = 0
for i in range(0,4):
    #row.append(i)
    for j in range(0,4):
        # if j not in column:
        #     column.append(j)
        if i==j:
            # row[k] = i 
            # column[k] = i
            # data[k] = 1    
            row.append(i)
            column.append(j)
            data.append(1)
        # else:
        #     data.append(0)    

arg_data = np.array(data)
arg_row = np.array(row)
arg_column = np.array(column)
print("arg_data: ",arg_data)
print("arg_row: ",arg_row)
print("arg_column: ",arg_column)
#csr_array_object:spmatrix = csr_matrix((row, column, data),shape=(3, 3))
#csr_array_object:spmatrix = csr_matrix(data, (row, column))
csr_array_object:spmatrix = csr_array((arg_data, (arg_row, arg_column)), shape=(4,4))
print("csr_array_object: ", csr_array_object)
print("csr_array_object.getcol(1): ", csr_array_object.getcol(1))


