import cv2
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import random

# Load the image
image_path = r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\newww\final\har.jpg'
image = cv2.imread(image_path)

# Load the Haar Cascade frontal face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Display the grey image
cv2.imshow("Gray image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Apply mean filtering with a kernel size of 5x5
filtered_image = cv2.blur(gray_image, (5, 5))

# Apply binary thresholding
_, binary_image = cv2.threshold(filtered_image, 127, 255, cv2.THRESH_BINARY)
# Display the binary image
cv2.imshow("Binary image", binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Morphological boundary extraction (Canny edge detection)
boundary_image = cv2.Canny(binary_image, 30, 150)
# Display the confused image
cv2.imshow("Boundary image", boundary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define the kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Perform morphological dilation to thicken the boundaries
thickened_image = cv2.dilate(boundary_image, kernel, iterations=1)

# Define the kernel for morphological closing
kernel = np.ones((5, 5), np.uint8)

# Perform morphological closing to fill the gaps
filled_image = cv2.morphologyEx(thickened_image, cv2.MORPH_CLOSE, kernel)

# Define the kernel for morphological closure
kernel = np.ones((5, 5), np.uint8)

# Perform morphological dilation
dilated_image = cv2.dilate(filled_image, kernel, iterations=1)

# Perform morphological erosion
closed_image = cv2.erode(dilated_image, kernel, iterations=1)

# Define the kernel for morphological erosion
kernel = np.ones((3, 3), np.uint8)

# Perform morphological erosion to apply lateral erosion
eroded_image = cv2.erode(closed_image, kernel, iterations=1)
# Display the confused image
cv2.imshow("Eroded image", eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perform face detection on the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Function to compute MD5 hash
def calculate_md5(string):
    encoded_string = string.encode('utf-8')
    md5_hash = hashlib.md5(encoded_string).hexdigest()
    return md5_hash

# Function to generate chaotic sequences S1 and S2 using 2D-DMHM chaotic system
def generate_chaotic_sequences12(theta, a_0, b_0, H, W):
    # Initialize sequences S1 and S2
    S1 = np.zeros((H, W))
    S2 = np.zeros((H, W))
   
    # Initial conditions
    x = theta
    y = a_0
   
    # Iterate over each pixel position in the image
    for i in range(H):
        for j in range(W):
            # Update equations for 2D-DMHM chaotic system
            x_next = x * np.sin(a * (1 - x)) + k * np.sin(y) * x
            y_next = x + y
           
            # Store the values in the sequences
            S1[i, j] = x_next
            S2[i, j] = y_next
           
            # Update for the next iteration
            x = x_next
            y = y_next
   
    return S1, S2

# Function to generate 𝑆𝑅𝐺𝐵 and 𝑆𝑏𝑖𝑡 from chaotic sequences S1 and S2
def generate_chaotic_sequencesRZ(S1, S2, H, W):
    # Generate 𝑆𝑅𝐺𝐵 based on S1
    S_RGB = np.floor((S1 * (H * W) * 1015) % 6).astype(int)

    # Generate 𝑆𝑏𝑖𝑡 based on S2
    S_bit = np.floor((S2 * 1015) % 256).astype(int)

    return S_RGB, S_bit

# Function to perform RGB permutation
def rgb_permutation(P, SRGB):
    # Decompose P into its RGB channels
    P_R = P[:, :, 0]  # Red channel
    P_G = P[:, :, 1]  # Green channel
    P_B = P[:, :, 2]  # Blue channel
   
    # Get the dimensions of the image
    H, W, _ = P.shape
   
    # Initialize the permuted image
    C1 = np.zeros_like(P)
   
    # Apply permutation rules determined by SRGB
    for i in range(H):
        for j in range(W):
            # Determine permutation rules using SRGB
            permutation_index = SRGB[i, j]
           
            # Perform permutation
            if permutation_index == 0:
                C1[i, j] = [P_B[i, j], P_R[i, j], P_G[i, j]]
            elif permutation_index == 1:
                C1[i, j] = [P_B[i, j], P_G[i, j], P_R[i, j]]
            elif permutation_index == 2:
                C1[i, j] = [P_R[i, j], P_B[i, j], P_G[i, j]]
            elif permutation_index == 3:
                C1[i, j] = [P_G[i, j], P_B[i, j], P_R[i, j]]
            elif permutation_index == 4:
                C1[i, j] = [P_R[i, j], P_G[i, j], P_B[i, j]]
            else:  # permutation_index == 5
                C1[i, j] = [P_G[i, j], P_R[i, j], P_B[i, j]]
   
    return C1

def zigzag_scramble(image):
    height, width, channels = image.shape
    output = np.zeros((height, width, channels), dtype=np.uint8)
   
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                if (i+j) % 2 == 0:
                    output[i, j, c] = image[height-1-i, width-1-j, c]
                else:
                    output[i, j, c] = image[i, j, c]
   
    return output

# Function to generate chaotic sequences S1 and S2 using 2D-DMHM chaotic system
def generate_chaotic_sequences3(theta, a_0, b_0, H, W):
    # Initialize sequences S1 and S2
    S3 = np.zeros((H, W))
    
   
    # Initial conditions
    x = theta
    y = a_0
   
    # Iterate over each pixel position in the image
    for i in range(H):
        for j in range(W):
            # Update equations for 2D-DMHM chaotic system
            x_next = x * np.sin(a * (1 - x)) + k * np.sin(y) * x
            
           
            # Store the values in the sequences
            S3[i, j] = x_next
            
           
            # Update for the next iteration
            x = x_next
        
   
    return S3

def generate_S4(S3, H, W):
    # Generate 𝑆4 sequence
    S4 = S3*(H*W+19*H*W)
    return S4

def generate_Sk(S3, H, W):
    # Ensure inputs are converted to NumPy arrays
    S3 = np.array(S3)
    H = np.array(H)
    W = np.array(W)
    
    # Generate 𝑆𝑘 sequence
    Sk = np.int_((S3 * (H * W) * (10**15)) % 256)
    return Sk

def reconstruct_key_matrix(Sk):
    # Get the shape of Sk
    sk_shape = Sk.shape
    
    # Calculate the total number of elements in Sk
    total_elements = np.prod(sk_shape)
    
    # Infer the shape of the key matrix by finding a suitable square root
    key_shape = (int(np.sqrt(total_elements)), int(np.sqrt(total_elements)))
    
    # Reshape the preprocessed sequence Sk into the shape of the key matrix K
    K = np.reshape(Sk, key_shape)
    
    # Optionally, apply any additional transformations or operations to finalize K
    
    return K

def count_rows_columns(matrix):
    if isinstance(matrix, list):
        if isinstance(matrix[0], list):
            # If matrix is a list of lists (2D)
            num_rows = len(matrix)
            num_columns = len(matrix[0]) if matrix else 0
        else:
            # If matrix is a list of lists of lists (3D)
            num_rows = len(matrix)
            num_columns = len(matrix[0]) if matrix else 0
    elif isinstance(matrix, np.ndarray):
        # If matrix is a numpy array
        if matrix.ndim == 2:
            # If matrix is 2D
            num_rows, num_columns = matrix.shape
        elif matrix.ndim == 3:
            # If matrix is 3D
            num_layers, num_rows, num_columns = matrix.shape
            return num_rows, num_columns  # Return only num_rows and num_columns
    else:
        raise ValueError("Unsupported data type. Please provide a list of lists or a numpy array.")

    return num_rows, num_columns  # Return a tuple containing num_rows and num_columns

def generate_square_matrix(C2, size):
    # Read the image
    image = C2
    
    # Convert to grayscale if the image is color
    if len(image.shape) > 2:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image
    
    # Resize the image to make it square
    resized_image = cv2.resize(grayscale_image, (size, size))
    
    # Convert the resized image to a 2D numpy array
    square_matrix = np.array(resized_image)
    
    return square_matrix

def extend_to_bit_level(matrix):
    # Convert the matrix to binary representation
    binary_matrix = np.unpackbits(matrix.astype(np.uint8), axis=-1)
    
    # Determine the maximum length between the binary representations
    max_length = binary_matrix.shape[-1]
    
    # Pad the binary representations to ensure they have equal length
    padded_matrix = np.pad(binary_matrix, ((0, 0),) * (binary_matrix.ndim - 1) + ((0, max_length - binary_matrix.shape[-1]),), mode='constant')
    
    return padded_matrix

def encode_dna(bit, se_value):
    # Define the DNA coding rules
    coding_rules = {
        0: [0, 0, 0, 0],
        1: [0, 0, 0, 1],
        2: [0, 0, 1, 0],
        3: [0, 0, 1, 1],
        4: [0, 1, 0, 0],
        5: [0, 1, 0, 1],
        6: [0, 1, 1, 0],
        7: [0, 1, 1, 1],
    }
    # If the bit value exceeds 7, use modulo operation to handle it
    return coding_rules[bit % 8]

def dna_encode(matrix, SE):
    # Initialize an empty list to store DNA-coded sequences
    dna_sequences = []
    # Find the maximum dimension (rows or columns)
    max_dim = max(matrix.shape)
    # Iterate over each row of the matrix
    for i in range(matrix.shape[0]):
        # Pad the row with zeros to make it a square matrix
        padded_row = np.pad(matrix[i], (0, max_dim - matrix.shape[1]), mode='constant')
        # Encode the values using DNA coding rules and SE sequence
        dna_row = []
        for value, se_value in zip(padded_row, SE[i]):
            dna_code = encode_dna(value, se_value)
            dna_row.extend(dna_code)
        # Append the DNA-coded row to the list of DNA-coded sequences
        dna_sequences.append(dna_row)
    # Convert the list of DNA-coded sequences into a numpy array
    dna_matrix = np.array(dna_sequences)
    return dna_matrix

def adjust_matrix_dimensions1(C2_prime, K):
    # Get the shape of the matrices
    rows_c2, cols_c2 = C2_prime.shape
    rows_k, cols_k = K.shape
    
    # If the number of columns in C2_prime is greater, extend K horizontally
    if cols_c2 > cols_k:
        num_cols_to_add = cols_c2 - cols_k
        K_extended = np.pad(K, ((0, 0), (0, num_cols_to_add)), mode='constant')
    else:
        K_extended = K
    
    return K_extended

def sub_dna_matrices(C2_prime, K):
    # Compute the dimensions based on the specified formula
    i = 2 * C2_prime.shape[0]
    j = 14 * C2_prime.shape[1]
    
    # Initialize an empty array to store the result
    result = np.zeros((i, j), dtype=np.int32)
    
    # Iterate over each row and column of the matrices
    for row_idx in range(C2_prime.shape[0]):
        for col_idx in range(C2_prime.shape[1]):
            # Compute the corresponding indices in the result matrix
            result_row_idx = 2 * row_idx
            result_col_idx = 14 * col_idx
            
            # Check if the indices are within the bounds of the result matrix
            if result_row_idx < i and result_col_idx < j:
                # Subtract the corresponding values from C2_prime and K matrices
                result[result_row_idx, result_col_idx] = C2_prime[row_idx, col_idx] - K[row_idx, col_idx]
    
    return result

def adjust_matrix_dimensions(matrix, target_shape):
    # Create a new matrix with the target shape filled with zeros
    padded_matrix = np.zeros(target_shape, dtype=matrix.dtype)
    # Get the dimensions of the original matrix
    orig_rows, orig_cols = matrix.shape
    # Compute the starting row and column indices for copying elements
    start_row = (target_shape[0] - orig_rows) // 2
    start_col = (target_shape[1] - orig_cols) // 2
    # Copy elements from the original matrix to the appropriate positions in the padded matrix
    padded_matrix[start_row:start_row+orig_rows, start_col:start_col+orig_cols] = matrix
    return padded_matrix

def add_dna_matrices(matrix1, matrix2):
    # Adjust the dimensions of matrix2 to match matrix1
    target_shape = matrix1.shape
    adjusted_matrix2 = adjust_matrix_dimensions(matrix2, target_shape)
    # Perform the addition operation between the matrices
    result_matrix = np.add(matrix1, adjusted_matrix2)
    return result_matrix

def chain_encrypt(matrix_C2):
    # Extract dimensions of the matrix
    height, width = matrix_C2.shape
    
    # Initialize the resulting ciphertext matrix
    ciphertext = np.zeros((height, width), dtype=int)
    
    # Iterate over each layer (R, G, B)
    for layer in range(3):
        # Find the last digit of the DNA encoding in the lower right corner of the current layer
        last_digit = matrix_C2[height-1, width-1] % 10
        
        # Encrypt the first digit of the subsequent layer using the last digit of the current layer
        next_layer = (layer + 1) % 3
        ciphertext[0, 0] = (matrix_C2[0, 0] + last_digit) % 10
        
        # Iterate over the remaining pixels in the current layer
        for i in range(height):
            for j in range(width):
                # Skip the lower right corner pixel
                if i == height-1 and j == width-1:
                    continue
                
                # Encrypt the pixel using the last digit of the DNA encoding in the lower right corner
                ciphertext[i, j] = (matrix_C2[i, j] + last_digit) % 10
        
        # Update the last digit for the next layer
        last_digit = ciphertext[height-1, width-1] % 10
    
    return ciphertext

def discontinuos_diffusion(image, num_swaps):

    b_image = np.copy(image)
    height, width, _ = b_image.shape
    for _ in range(num_swaps):
        # Generate random pixel coordinates
        x1, y1 = random.randint(0, width - 1), random.randint(0, height - 1)
        x2, y2 = random.randint(0, width - 1), random.randint(0, height - 1)
        # Swap the pixel values (bitwise)
        b_image[y1, x1], b_image[y2, x2] = b_image[y2, x2], b_image[y1, x1]
    return b_image

def imshowImage(C4_3):
    image=C4_3
    ns=500000
    im2=discontinuos_diffusion(image,ns)
    return im2

def encrypt_decrypt(image, chaotic_sequence):
    """Encrypt or decrypt the image using the chaotic sequence."""
    height, width, channels = image.shape
    encrypted_image = np.zeros_like(image)
    
    # Loop through each pixel position (i, j)
    for i in range(height):
        for j in range(width):
            # Determine the current chaotic sequence value
            current_value = chaotic_sequence[i, j]
            
            # Determine the previous pixel value based on the position (i, j)
            previous_pixel_value = 0
            if i == 0 and j == 0:
                previous_pixel_value = chaotic_sequence[-1, -1]
            elif i == 0 and j != 0:
                previous_pixel_value = chaotic_sequence[-1, j - 1]
            elif i != 0:
                previous_pixel_value = chaotic_sequence[i - 1, j]
            
            # Loop through each channel
            for k in range(channels):
                # Compute the new pixel value based on the encryption/decryption formula
                new_pixel_value = (current_value + previous_pixel_value + image[i, j, k]) % 256
                
                # Assign the new pixel value to the encrypted/decrypted image
                encrypted_image[i, j, k] = new_pixel_value
                
    return encrypted_image

def discontinuous_diffusionR(C4):
    """Encrypt an image using discontinuous diffusion."""
    # Read the image
    image = C4

    # Encrypt the image
    encrypted_image = encrypt_decrypt(image, S_bit)
    
    return encrypted_image

def reconstruct_key_matrix(S_bit):
    # Assuming SBIT is a NumPy array representing the chaotic sequence
    # You may need to adjust the size and shape of the key matrix based on your requirements
    
    # Define the size of the key matrix (assuming it's a square matrix for simplicity)
    matrix_size = int(np.sqrt(S_bit.size))  # Adjust this based on the size of SBIT
    
    # Reshape the chaotic sequence into a square matrix
    reshaped_SBIT = S_bit [:matrix_size**2].reshape(matrix_size, matrix_size)
    
    # Use the reshaped chaotic sequence as the key matrix
    Kh = reshaped_SBIT
    
    return Kh

def bit_level_confusion(encrypted_image, Kh, s):
    # Load the image
    img = encrypted_image
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert image to numpy array
    C = np.array(img_gray)
    rec_code= 0
    C4_prime = np.zeros_like(C)

    # Bitwise XOR operation between C4 and Kh
    for i in range(len(C)):
        for j in range(len(C[0])):
            C4_prime[i][j] = C[i][j] ^ Kh[i][j]

    # Bit-level right shift by s bits
    C5 = np.zeros_like(C)
    for i in range(len(C)):
        for j in range(len(C[0])):
            new_index = i + s
            if new_index < len(C):
                C5[new_index][j] = C4_prime[i][j]
    C5 = rec_code
    # Return the resulting image matrix
    return img



# Iterate over detected faces
for i, (x, y, w, h) in enumerate(faces):
    # Extract the face region from the original image
    face = image[y:y+h, x:x+w]
   
    # Save the extracted face region as a separate image with a unique filename
    face_path = r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\newww\final\har_face_{}.jpg'.format(i)
    cv2.imwrite(face_path, face)

    # Display the detected face
    cv2.imshow('Detected Face {}'.format(i+1), face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    # Check if the image was loaded successfully
    if face is None:
        print("Error: Unable to load the image.")
        exit()

    # Resize the image to have square dimensions
    min_dim = min(face.shape[:2])
    image_resized = cv2.resize(face, (min_dim, min_dim))


    # Split the resized image into channels
    b, g, r = cv2.split(image_resized)

    # Compute eigenvalues for each channel
    eigenvalues_b = np.linalg.eigvals(b)
    eigenvalues_g = np.linalg.eigvals(g)
    eigenvalues_r = np.linalg.eigvals(r)

    # Combine eigenvalues from all channels
    eigenvalues = np.concatenate((eigenvalues_b, eigenvalues_g, eigenvalues_r))

    # Compute scrambling values h(x)
    h = [eigenvalues[i] for i in range(32)]  # Example: Take first 32 eigenvalues as scrambling values
    # Compute key parameters
    real_h = np.real(h)  # Convert complex numbers to real numbers
    theta = 0.1 + sum(real_h[0::3]) % 0.21
    a_0 = sum(real_h[1::3]) % 1.5
    b_0 = sum(real_h[2::3]) % 2.4

    # Calculate MD5 hashes
    theta_md5 = calculate_md5(str(theta))
    a_0_md5 = calculate_md5(str(a_0))
    b_0_md5 = calculate_md5(str(b_0))

    print("MD5 hash of theta:", theta_md5)
    print("MD5 hash of a_0:", a_0_md5)
    print("MD5 hash of b_0:", b_0_md5)

    # Parameters for the 2D-DMHM chaotic system
    a = 0.1
    k = 1.72

    # Check if the image was loaded successfully
    if image_resized is None:
        print("Error: Unable to load the image.")
        exit()

    # Get the dimensions of the image
    H, W, _ = image_resized.shape

    # Convert MD5 hash values to numeric values
    theta = int(theta_md5, 16) % 0.21
    a_0 = int(a_0_md5, 16) % 1.5
    b_0 = int(b_0_md5, 16) % 2.4

    # Generate chaotic sequences S1 and S2
    S1, S2 = generate_chaotic_sequences12(theta, a_0, b_0, H, W)

    # Print the generated sequences (you may want to visualize them)
    print("S1:")
    print(S1)
    print("\nS2:")
    print(S2)

    # Example usage:
    # Assuming S1 and S2 are already defined
    # H, W represent the dimensions of the image

    # Generate 𝑆𝑅𝐺𝐵 and 𝑆𝑏𝑖𝑡 sequences
    S_RGB, S_bit = generate_chaotic_sequencesRZ(S1, S2, H, W)

    # Reshape S_RGB to match the dimensions of the input image
    #S_RGB = S_RGB.reshape((H, W))

    print('SRGB:',S_RGB)
    print("SBIT: ",S_bit)

    # Load the image using OpenCV
    P = image_resized

    # Convert the image to RGB format
    P = cv2.cvtColor(P, cv2.COLOR_BGR2RGB)

    # Generate a random SRGB matrix for demonstration
    SRGB = np.random.randint(0, 6, size=P.shape[:2])

    # Perform RGB permutation
    C1 = rgb_permutation(P, SRGB)

    # Display the permuted image
    plt.imshow(C1)
    plt.axis('off')
    plt.title('Permuted Image')
    plt.show()

    cv2.imwrite(r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\rgb_img.jpg', C1)

    # Example usage:
    C2 = zigzag_scramble(C1)

    # Display the permuted image
    plt.imshow(C2)
    plt.axis('off')
    plt.title('ZigZag Scrambled Image')
    plt.show()

    cv2.imwrite(r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\zigzag_img.jpg', C2)

    # Check if the image was loaded successfully
    if C2 is None:
        print("Error: Unable to load the image.")
        exit()

    # Resize the image to have square dimensions
    min_dim = min(C2.shape[:2])
    image_resized = cv2.resize(C2, (min_dim, min_dim))


    # Split the resized image into channels
    b, g, r = cv2.split(image_resized)

    # Compute eigenvalues for each channel
    eigenvalues_b = np.linalg.eigvals(b)
    eigenvalues_g = np.linalg.eigvals(g)
    eigenvalues_r = np.linalg.eigvals(r)

    # Combine eigenvalues from all channels
    eigenvalues = np.concatenate((eigenvalues_b, eigenvalues_g, eigenvalues_r))

    # Compute scrambling values h(x)
    new_h = [eigenvalues[i] for i in range(32)]  # Example: Take first 32 eigenvalues as scrambling values
    # Compute key parameters
    new_real_h = np.real(h)  # Convert complex numbers to real numbers
    new_theta = 0.1 + sum(real_h[0::3]) % 0.21
    new_a_0 = sum(real_h[1::3]) % 1.5
    new_b_0 = sum(real_h[2::3]) % 2.4

    # Calculate MD5 hashes
    new_theta_md5 = calculate_md5(str(theta))
    new_a_0_md5 = calculate_md5(str(a_0))
    new_b_0_md5 = calculate_md5(str(b_0))

    print("MD5 hash of theta:",new_theta_md5)
    print("MD5 hash of a_0:", new_a_0_md5)
    print("MD5 hash of b_0:", new_b_0_md5)

    # Parameters for the 2D-DMHM chaotic system
    a = 0.1
    k = 1.72

    # Check if the image was loaded successfully
    if image_resized is None:
        print("Error: Unable to load the image.")
        exit()

    # Get the dimensions of the image
    H, W, _ = image_resized.shape

    # Convert MD5 hash values to numeric values
    new_theta = int(theta_md5, 16) % 0.21
    new_a_0 = int(a_0_md5, 16) % 1.5
    new_b_0 = int(b_0_md5, 16) % 2.4

    # Generate chaotic sequences S3
    S3 = generate_chaotic_sequences3(new_theta, new_a_0, new_b_0, H, W)

    # Print the generated sequences (you may want to visualize them)
    print("S3:")
    print(S3)

    # Generate 𝑆4 sequence
    S4 = generate_S4(S3, H, W)
    print("S4:", S4)

    Sk = generate_Sk(S3, H, W)
    print("Sequence Sk:")
    print(Sk)

    #SE = generate_SE(S4, H, W)
    SE = np.floor((S4 * (14 * H * W) * (10**15)) % 8) + 1
    print("SE:", SE)

    # Generate sequence SD
    SD = np.floor((S4 * (4 * H * W + 18 * H * W) * (10**15)) % 8) + 1
    print("SD:", SD)

    K = reconstruct_key_matrix(Sk)
    print("K matrix: ",K)

    rows_K, columns_K= count_rows_columns(K)
    print("Number of rows of K:", rows_K)
    print("Number of columns of K:", columns_K)

    # Specify the path to the image and the desired size of the square matrix
    matrix_size = rows_K  # Adjust this according to your preference

    # Generate the square matrix from the image
    C2_square_matrix = generate_square_matrix(C2, matrix_size)

    # Display or further process the square matrix as needed
    print("Square matrix shape:", C2_square_matrix.shape)

    print("C2 matrix:",C2_square_matrix)

    # Extend the matrix to bit level
    extended_matrix_C2 = extend_to_bit_level(C2_square_matrix)
    extended_matrix_K = extend_to_bit_level(K)
    # Print the original and extended matrices
    print("\nBit level Extended C2 Matrix:")
    print(extended_matrix_C2)
    print("\nBit level Extended K Matrix:")
    print(extended_matrix_K)

    rows, columns = count_rows_columns(extended_matrix_K)
    print("Number of rows of K_bits:", rows)
    print("Number of columns of K_bits:", columns)

    rows, columns = count_rows_columns(extended_matrix_C2)
    print("Number of rows of C2_bits:", rows)
    print("Number of columns of C2_bits:", columns)

    # Encode C2' using the provided formula and coding sequence
    C2_prime = dna_encode(extended_matrix_C2, SE)
    K_prime = dna_encode(extended_matrix_K, SE)
    print("C_prime: ",C2_prime)
    print("K_prime: ",K_prime)

    rows, columns = count_rows_columns(C2_prime)
    print("Number of rows of C2_prime:", rows)
    print("Number of columns of C2_prime:", columns)

    rows, columns = count_rows_columns(K_prime)
    print("Number of rows of K_prime:", rows)
    print("Number of columns of K_prime:", columns)

    # Example usage:
    # Assuming C2_prime and K are numpy arrays representing matrices
    # Replace the values of C2_prime and K with your actual data

    # Adjust the dimensions of the K matrix
    K_adjusted = adjust_matrix_dimensions1(C2_prime, K)
    rows, columns = count_rows_columns(K_adjusted)
    print("Number of rows of K_adjusted:", rows)
    print("Number of columns of K_adjusted:", columns)

    # Compute the subtraction of DNA-encoded matrices
    Cd = sub_dna_matrices(C2_prime, K_adjusted)

    # Print the resulting matrix
    print("Cd Matrix:")
    print(Cd)

    rows, columns = count_rows_columns(Cd)
    print("Number of rows of Cd:", rows)
    print("Number of columns of Cd:", columns)

    # Compute the adjusted matrix
    adjusted_K_matrix2 = adjust_matrix_dimensions(K, Cd.shape)

    # Perform the XOR operation between the adjusted matrices
    Cd_prime = np.bitwise_xor(Cd, adjusted_K_matrix2)

    # Print the resulting matrix
    print("Cd_prime Matrix:")
    print(Cd_prime)

    rows, columns = count_rows_columns(Cd_prime)
    print("Number of rows of Cd_prime:", rows)
    print("Number of columns of Cd_prime", columns)

    # Compute the result matrix
    C2_double_prime = add_dna_matrices(Cd_prime, C2_prime)

    # Print the resulting matrix
    print("C2_double_prime Matrix:")
    print(C2_double_prime)

    C3 = chain_encrypt(C2_double_prime)
    #print("cipher text: ",C3)

    # Number of pixel swaps to perform (controls the degree of blurring)
    num_swaps = 100000  # Adjust this value based on the desired level of blurring

    # Blur the image by swapping pixels bit by bit
    C4_prime= discontinuos_diffusion(C2,num_swaps) 

    # Display the original and blurred images
    cv2.imshow("Image after DNA_Encryption", C4_prime)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\DNA_img.jpg', C4_prime)

    # Encrypt the image using discontinuous diffusion
    diffusion_image = discontinuous_diffusionR(C4_prime) #diffusion
    

    # Display the encrypted image
    cv2.imshow('Diffusion Image', diffusion_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    C4 = imshowImage(diffusion_image)
    cv2.imwrite(r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\neww\final\diffusion.jpg', diffusion_image)

    s=300
    # Example usage:
    # Specify the image path
    # Load Kh and s values accordingly
    # After this step, C will be updated to C5
    KH = reconstruct_key_matrix(S_bit)
    print("KH matrix:",KH)
    C5 = bit_level_confusion(C4, KH, s)

    # Display the encrypted image
    cv2.imshow('Bit level', C5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\Confusion_final.jpg',C5)

