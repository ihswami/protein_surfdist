import numpy as np
import torch

def get_molecule_data_from_text_file(file_path: str) -> np.ndarray:
    """get x,y,z and properties from the text file and return as numpy array"""    
    with open(file_path, 'r') as file:
        file_content = file.readlines()
        data = [list(map(float, line.strip().split())) for line in file_content]
        molecule_data = np.array(data)
    return molecule_data

def perform_optimization(A, B, eigenvalues1, eigenvalues2, k, iterations=100000, consecutive_tol_count=1000):
    """Optimize matrices C and D."""
    if torch.isnan(A).any() or torch.isnan(B).any():
        print("NaN found in initial tensors A or B")
        print("A:", A)
        print("B:", B)
        
    if torch.isnan(eigenvalues1).any() or torch.isnan(eigenvalues2).any():
        print("NaN found in eigenvalues")
        print("Eigenvalues1:", eigenvalues1)
        print("Eigenvalues2:", eigenvalues2)
        
    eigenvalues1 = torch.tensor(eigenvalues1, dtype=torch.float32)  # Ensure conversion to float32
    eigenvalues2 = torch.tensor(eigenvalues2, dtype=torch.float32)  # Ensure conversion to float32
    
    C, D = initialize_matrices(A, B, k)
    
    optimizer = torch.optim.Adam([C, D], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    
    min_loss = float('inf')
    best_C, best_D = None, None
    loss_diff_count = 0  # Counter for consecutive small loss differences
    last_loss = None  # Store the last loss to calculate difference

    for i in range(iterations):
        optimizer.zero_grad()
        loss = cost_function(C, D, A, B, eigenvalues1, eigenvalues2, k)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        if last_loss is not None:
            if abs(last_loss - current_loss) <= 1e-4 and last_loss < 0.5:
                loss_diff_count += 1
            else:
                loss_diff_count = 0  # Reset counter if difference is greater

        if loss_diff_count >= consecutive_tol_count:
            print(f"Early stopping at iteration {i} due to small loss change for {consecutive_tol_count} consecutive iterations will last loss {last_loss}.")
            break

        last_loss = current_loss  # Update last_loss for the next iteration

        if current_loss < min_loss:
            min_loss = current_loss
            best_C, best_D = C.clone(), D.clone()

        if min_loss <= 1e-6:
            print(f"Early stopping at iteration {i} with loss {min_loss}")
            break
        
        if i % 10000 == 0:
            print(f"Iteration {i}: Loss = {current_loss}, LR = {scheduler.get_last_lr()[0]}")

    return best_C, best_D, min_loss


def cost_function(C, D, A, B, eigenvalues1, eigenvalues2, k, w_iso= 1):
    """Define the cost function to minimize."""
    diffC = torch.mm(C, A) - B
    diffD = torch.mm(D, B) - A
    isoC = torch.mm(C, torch.diag(eigenvalues1[:k])) - torch.mm(torch.diag(eigenvalues2[:k]), C)
    isoD = torch.mm(D, torch.diag(eigenvalues2[:k])) - torch.mm(torch.diag(eigenvalues1[:k]), D)
    inverDC = torch.mm(D, C) - torch.eye(k, device=C.device)
    inverCD = torch.mm(C, D) - torch.eye(k, device=C.device)
    harmitianCD = torch.mm(C, D.T) - torch.eye(k, device=C.device)
    harmitianDC = torch.mm(D, C.T) - torch.eye(k, device=C.device)

    return (diffC.norm()**2 + diffD.norm()**2  + w_iso * isoC.norm()**2 + w_iso * isoD.norm()**2 + harmitianCD.norm()**2 + + harmitianDC.norm()**2)

def initialize_matrices(A, B, k):
    """
    Initializes matrices C and D based on checks calculated from A and B.

    Parameters:
        A (torch.Tensor): First matrix for comparison.
        B (torch.Tensor): Second matrix for comparison.
        k (int): Size of the identity matrix for calculations.
    
    Returns:
        tuple: (C, D, optimizer), where C and D are optimizable matrices, and optimizer is an Adam optimizer.
    """
    # Calculate check1, check2, and check3
    check1 = torch.sum(abs(torch.eye(k) @ A.clone().detach() - B.clone().detach()))
    check2 = torch.sum(abs(-1 * torch.eye(k) @ A.clone().detach() - B.clone().detach()))
    check3 =  torch.sum( abs(torch.tensor(create_custom_diagonal_matrix(k), dtype=torch.float32) @ A.clone().detach() - B.clone().detach()))
    check4 =  torch.sum( abs(-1 * torch.tensor(create_custom_diagonal_matrix(k), dtype=torch.float32) @ A.clone().detach() - B.clone().detach()))
    rand_dig = torch.tensor(generate_random_diagonal_matrix(k), dtype=torch.float32)
    check5 = torch.sum(abs(rand_dig @ A.clone().detach() - B.clone().detach()))

    # Determine which condition has the minimum check
    check = [check1, check2, check3, check4, check5]
    c_index = check.index(min(check))

    # Initialize matrices C and D based on the smallest check
    if c_index == 0:
        print("Using Unit Diagonal Matrix")
        C = torch.eye(k, k, requires_grad=True, dtype=torch.float32)  # Optimizable matrix
        D = torch.eye(k, k, requires_grad=True, dtype=torch.float32)  # Optimizable matrix
    
    elif c_index == 1:
        print("Using Negative Unit Diagonal Matrix")
        C = torch.tensor(-1 * torch.eye(k, k), requires_grad=True, dtype=torch.float32)  # Optimizable matrix
        D = torch.tensor(-1 * torch.eye(k, k), requires_grad=True, dtype=torch.float32)  # Optimizable matrix
    
    elif c_index == 2:
        print("Using First Half Elements as 1 and Another Half as -1 for a  Diagonal Matrix")
        C = torch.tensor(create_custom_diagonal_matrix(k), requires_grad=True, dtype=torch.float32)  # Optimizable matrix
        D = torch.tensor(create_custom_diagonal_matrix(k), requires_grad=True, dtype=torch.float32)  # Optimizable matrix
    
    elif c_index == 3:
        print("Using First Half Elements as -1 and Another Half as 1 for a  Diagonal Matrix")
        C = torch.tensor(-1 * create_custom_diagonal_matrix(k), requires_grad=True, dtype=torch.float32)  # Optimizable matrix
        D = torch.tensor(-1 * create_custom_diagonal_matrix(k), requires_grad=True, dtype=torch.float32)  # Optimizable matrix    
    else:
        print('Using a Random Matrix')
        C = torch.tensor(rand_dig, requires_grad=True, dtype=torch.float32)  # Optimizable matrix
        D = torch.tensor(rand_dig, requires_grad=True, dtype=torch.float32)  # Optimizable matrix
    
    return C, D

def create_custom_diagonal_matrix(size):
    """
    Create a square matrix with the first half of the diagonal elements set to 1,
    and the other half set to -1. All non-diagonal elements are zero.

    Parameters:
        size (int): The size of the square matrix (n x n).

    Returns:
        np.ndarray: The resulting matrix with specified diagonal pattern.
    """
    # Initialize a square matrix of the specified size with zeros
    matrix = np.zeros((size, size))

    # Determine the split point for the diagonal
    half_point = size // 2

    # Set the first half of the diagonal to 1
    for i in range(half_point):
        matrix[i, i] = 1

    # Set the second half of the diagonal to -1
    for i in range(half_point, size):
        matrix[i, i] = -1

    return matrix

def generate_random_diagonal_matrix(size, probabilities=None):
    """
    Generates a random square matrix with diagonal elements chosen from (1, -1).
    
    Parameters:
        size (int): The size of the square matrix.
        probabilities (list or tuple, optional): The probabilities of selecting 1 or -1 for the diagonal.
            If None, the probability for each is equal (0.5, 0.5).
    
    Returns:
        np.ndarray: The generated random matrix with the specified diagonal.
    """
    # If probabilities aren't provided, use equal probability
    if probabilities is None:
        probabilities = [0.5, 0.5]
    
    # Randomly choose diagonal elements from 1 or -1 based on probabilities
    diagonal_elements = np.random.choice([1, -1], size=size, p=probabilities)
    
    # Initialize an n x n matrix with zeros
    random_matrix = np.zeros((size, size))
    
    # Assign the diagonal elements to the matrix
    np.fill_diagonal(random_matrix, diagonal_elements)
    
    return random_matrix



