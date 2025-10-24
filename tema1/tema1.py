import pathlib
import math

# exercise 1
def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    
    """ Takes a file as input and returns the matrix and vector corresponding to the system of equations in the file"""
    
    A = []
    B = []
    variables = ["x", "y", "z"]
    file = open(path, 'r')
    for line in file:
        left, right = line.split('=')
        coefficients = []
        left = left.replace(" + ", " +").replace(" - ", " -").split()
        for var in variables:
            value = 0
            for part in left:
                if var in part:
                    num = part.replace(var, "")
                    if num == "" or num == "+":
                        num = "1"
                    elif num == "-":
                        num = "-1"
                    value = float(num)
                    break
            coefficients.append(value)
        A.append(coefficients)
        B.append(float(right.strip()))
    file.close()
    return A, B

# exercise 2.1
def determinant(matrix: list[list[float]]) -> float:
    
    """Computes the determinant of a matrix"""

    determinant = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                   matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                   matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])) 
    return determinant

# exercise 2.2
def trace(matrix: list[list[float]]) -> float:

    """Computes the trace of a matrix"""

    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    return trace

# exercise 2.3
def norm(vector: list[float]) -> float:
    
    """Computes the norm of a vector"""
    
    norm = 0.0
    for element in vector:
        norm = norm + element ** 2
    norm = math.sqrt(norm)
    return norm

# exercise 2.4
def transpose(matrix: list[list[float]]) -> list[list[float]]:

    """Computes the transpose of a given matrix"""
    
    transpose = [[0.0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for line in range(0, len(matrix)):
        for col in range(0, len(matrix[0])):
            transpose[col][line] = matrix[line][col]
    return transpose

# exercise 2.5
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    
    """Computes the multiplication between a matrix and a vector"""

    product = []
    for line in range(0, len(matrix)):
        element = 0
        for col in range(0, len(matrix[0])):
            element = element + matrix[line][col] * vector[col]
        product.append(element)
    return product

# exercise 3
def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:

    """Solves a system of linear equations using Cramer's rule"""

    solution = []
    det_matrix = determinant(matrix=matrix)
    
    if det_matrix == 0:
        raise ValueError("Matrix determinant is zero — system has no unique solution")
    
    for var in range(0, 3):
        matrix_var = [[matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]
        for i in range(0, 3):
            matrix_var[i][var] = vector[i]
        det_matrix_var = determinant(matrix=matrix_var)
        result = det_matrix_var / det_matrix
        solution.append(result)
    return solution

# exercise 4
def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:

    """Computes the minor M_ij for a given matrix and given i and j"""

    minor = [[0.0 for _ in range(0, 2)] for _ in range(0, 2)]
    line = 0
    col = 0
    for matrix_line in range(0, len(matrix)):
        if matrix_line == i:
            continue
        for matrix_col in range(0, len(matrix[0])):
            if matrix_col == j:
                continue
            minor[line][col] = matrix[matrix_line][matrix_col]
            if col == 1:
                col = 0
                line = 1
            else:
                col = 1 
    return minor

def cofactor(matrix: list[list[float]]) -> list[list[float]]:

    """Computes the cofactor matrix of a given matrix"""

    cofactor = [[0.0 for _ in range(0, len(matrix[0]))] for _ in range(0, len(matrix))]
    for line in range(0, len(matrix)):
        for col in range(0, len(matrix)):
            m = minor(matrix, line, col)
            cofactor[line][col] = ((-1.0) ** (line + col)) * (m[0][0] * m[1][1] - m[0][1] * m[1][0])
    return cofactor

def adjoint(matrix: list[list[float]]) -> list[list[float]]:

    """Computes the adjoint matrix of a matrix"""

    return transpose(cofactor(matrix=matrix))

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:

    """Solves a system of linear equations using Inversion"""

    adj = adjoint(matrix=matrix)
    inverse = [[0.0 for _ in range(0, len(matrix[0]))] for _ in range(0, len(matrix))]
    det = determinant(matrix=matrix)
    for line in range(0, len(matrix)):
        for col in range(0, len(matrix[0])):
            inverse[line][col] = adj[line][col] / det
    return multiply(inverse, vector=vector)

# Verifications
A, B = load_system(pathlib.Path("input.txt"))
print(f"{A=} {B=}")
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")
print(f"{solve_cramer(A, B)=}")
print(f"{solve(A, B)=}")