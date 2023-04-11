import numpy as np

class Matrix:
    def __init__(self, cols=None, rows=None):
        if cols is None:
            cols = np.random.randint(1, 31)
        if rows is None:
            rows = np.random.randint(1, 31)
        self.cols = cols
        self.rows = rows
        self.values = np.random.rand(rows, cols)
        if cols == rows:
            self.det = np.linalg.det(self.values)
        else:
            self.det = None
        self.shape = (rows, cols)

    def rank(self):
        return float(np.linalg.matrix_rank(self.values))

    def trace(self):
        return np.trace(self.values)

    def determinant(self):
        if self.det is None:
            return "The matrix is not square, determinant cannot be calculated"
        return self.det

    def inverse(self):
        try:
            return str(np.linalg.inv(self.values))
        except np.linalg.LinAlgError:
            return "The matrix cannot be inverted"
        
    def eigenvalues_transpose(self):
        A = np.dot(self.values, self.values.T)
        B = np.dot(self.values.T, self.values)
        return {'eigenvalues A*A.T':str(np.linalg.eigvals(A)),'eigenvalues A.T*A':str(np.linalg.eigvals(B))}
