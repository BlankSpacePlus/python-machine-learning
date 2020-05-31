import numpy as np
from scipy import sparse

# 创建一个一维数组表示一个行向量
vector_row = np.array([1, 2, 3])

# 创建一个一维数组表示一个列向量
vector_column = np.array([[1], [2], [3]])

# 创建一个二维数组表示一个矩阵
matrix1 = np.array([[1, 2], [1, 2], [1, 2]])

# 利用Numpy内置矩阵数据结构
matrix1_object = np.mat([[1, 2], [1, 2], [1, 2]])

# 创建一个新的矩阵
matrix2 = np.array([[0, 0], [0, 1], [3, 0]])

# 创建一个压缩的稀疏行(CSR)矩阵
matrix2_sparse = sparse.csc_matrix(matrix2)

# 查看稀疏矩阵
print(matrix2_sparse)

# 创建一个更大的矩阵
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# 创建一个CSR矩阵
matrix_large_sparse = sparse.csr_matrix(matrix_large)

# 查看更大的稀疏矩阵
print(matrix_large_sparse)

# 创建一个行向量
vector = np.array([1, 2, 3, 4, 5, 6])

# 创建矩阵
matrix_vector = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 选择向量的第三个元素
print(vector[2])

# 选择第二行第二列
print(matrix_vector[1, 1])

# 选取一个向量的所有元素
print(vector[:])

# 选取从0开始一直到第3个（包含第3个）元素
print(vector[:3])

# 选取第3个元素之后的全部元素
print(vector[3:])

# 选取最后一个元素
print(vector[-1])

# 选取矩阵的第1行和第2行以及所有列
print(matrix_vector[:2, :])

# 选取所有行以及第2列
print(matrix_vector[:, 1:2])

# 选取所有行以及第2列并转换成一个新的行向量
print(matrix_vector[:, 1])

# 创建新的矩阵
matrix3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 查看行数和列数
print(matrix3.shape)

# 查看元素数量
print(matrix3.size)

# 查看维数
print(matrix3.ndim)

# 下面使用的矩阵是matrix_vector
# 创建一个匿名函数，返回输入值加上100以后的值
add_100 = lambda i: i+100

# 创建向量转化函数
vectorized_add_100 = np.vectorize(add_100)

# 对矩阵的所有元素应用这个函数
print(vectorized_add_100(matrix_vector))

# 用后矩阵本身不变
print(matrix_vector)

# 连续使用
print(vectorized_add_100(vectorized_add_100(matrix_vector)))

# 返回最大的元素
print(np.max(matrix_vector))

# 返回最小元素
print(np.min(matrix_vector))

# 找到每一列的最大元素
print(np.max(matrix_vector, axis=0))

# 找到每一行最大的元素
print(np.max(matrix_vector, axis=1))

# 返回平均值
print(np.mean(matrix_vector))

# 返回方差
print(np.var(matrix_vector))

# 返回标准差
print(np.std(matrix_vector))

# 求每一列的平均值
print(np.mean(matrix_vector, axis=0))

# 求每一行的方差
print(np.var(matrix_vector, axis=1))

# 将matrix3矩阵变为2×6矩阵
matrix4 = matrix3.reshape(2, 6)
print(matrix4)

# 上面的变形要求前后元素个数相同，且不会改变元素个数
print(matrix4.size)

# reshape时传入参数-1意味着可以根据需要填充元素
print(matrix3.reshape(1, -1))

# reshape如果提供一个整数，那么reshape会返回一个长度为该整数值的一维数组
print(matrix3.reshape(12))

# 转置matrix_vector矩阵
print(matrix_vector.T)

# 严格地讲，向量是不能被转置的
print(vector.T)

# 转置向量通常指二维数组表示形式下将行向量转换为列向量或者反向转换
print(np.array([[1, 2, 3, 4, 5, 6]]).T)

# 将matrix_vector矩阵展开
print(matrix_vector.flatten())

# 将矩阵展开的另一种策略是利用reshape创建一个行向量
print(matrix_vector.reshape(1, -1))

# 创建用于求秩的新矩阵
matrix5 = np.array([[1, 1, 1], [1, 1, 10], [1, 1, 15]])

# 计算矩阵matrix5的秩
print(np.linalg.matrix_rank(matrix5))

# 创建用于行列式求解的新矩阵
matrix6 = np.array([[1, 2, 3], [2, 4, 6], [3, 8, 9]])

# 求解矩阵matrix6的行列式
print(np.linalg.det(matrix6))

# 返回矩阵的对角线元素
print(matrix6.diagonal())

# 返回主对角线向上偏移量为1的对角线元素
print(matrix6.diagonal(offset=1))

# 返回主对角线向下偏移量为1的对角线元素
print(matrix6.diagonal(offset=-1))

# 返回矩阵的迹
print(matrix6.trace())

# 求迹的另外的方法（返回对角线元素并求和）
print(sum(matrix6.diagonal()))

# 创建一个求解特征值、特征向量的矩阵
matrix7 = np.array([[1, -1, 3], [1, 1, 6], [3, 8, 9]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(matrix7)

# 查看特征值
print(eigenvalues)

# 查看特征向量
print(eigenvectors)

# 构造两个点积（数量积）所需向量
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# 计算点积
print(np.dot(vector_a, vector_b))

# Python 3.5+ 版本可以这样求解点积
print(vector_a @ vector_b)

# 构造两个可用于加减的矩阵
matrix_a = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 2]])
matrix_b = np.array([[1, 3, 1], [1, 3, 1], [1, 3, 8]])

# 两矩阵相加
print(np.add(matrix_a, matrix_b))

# 两矩阵相减
print(np.subtract(matrix_a, matrix_b))

# 直接用+/-也可以做矩阵加减
print(matrix_a + matrix_b)
print(matrix_a - matrix_b)

# 构造两个可用于乘法的小矩阵
matrix_c = np.array([[1, 1], [1, 2]])
matrix_d = np.array([[1, 3], [1, 2]])

# 两矩阵相乘
print(np.dot(matrix_c, matrix_d))

# Python 3.5+ 版本可以这样求解矩阵乘法
print(matrix_c @ matrix_d)

# 我们也可以把两矩阵对应元素相乘，而非矩阵乘法
print(matrix_c * matrix_d)

# 创建一个用于求逆的矩阵
matrix8 = np.array([[1, 4], [2, 5]])

# 计算矩阵的逆
print(np.linalg.inv(matrix8))

# 验证一个矩阵和它的逆矩阵相乘等于I（单位矩阵）
print(matrix8 @ np.linalg.inv(matrix8))
