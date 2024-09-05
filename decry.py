import cv2
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import merge # type: ignore
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

def inverse_bit_level_confusion(encrypted_image, Kh, s):
    # Convert the encrypted image to a numpy array
    C5 = np.array(encrypted_image)

    # Check if the conversion was successful
    if C5.size == 0:
        raise ValueError("Input encrypted_image is not a valid numpy array")

    # Create an empty numpy array to store the decrypted image
    C4_prime = np.zeros_like(C5)

    # Bit-level left shift by s bits
    for i in range(len(C5)):
        for j in range(len(C5[0])):
            original_index = i - s
            if original_index >= 0:
                C4_prime[i][j] = C5[original_index][j]
            else:
                C4_prime[i][j] = C5[original_index + len(C5)][j]

    # Bitwise XOR operation between C4_prime and Kh
    decryted_image= merge.diffusion_image
    C = np.zeros_like(C4_prime)
    for i in range(len(C4_prime)):
        for j in range(len(C4_prime[0])):
            C[i][j] = C4_prime[i][j] ^ Kh[i][j]

    # Convert the decrypted numpy array back to an image
    decrypted_image = cv2.normalize(C, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return decryted_image

def decrypt_encrypt(encrypted_image, chaotic_sequence):
    """Decrypt or encrypt the image using the chaotic sequence."""
    height, width, channels = encrypted_image.shape
    decrypted_image = np.zeros_like(encrypted_image)
    
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
                # Compute the original pixel value based on the decryption/encryption formula
                original_pixel_value = (encrypted_image[i, j, k] - current_value - previous_pixel_value) % 256
                
                # Assign the original pixel value to the decrypted/encrypted image
                decrypted_image[i, j, k] = original_pixel_value
                
    return decrypted_image

def chain_decrypt(ciphertext):
    # Extract dimensions of the matrix
    height, width = ciphertext.shape
    
    # Initialize the resulting plaintext matrix
    plaintext = np.zeros((height, width), dtype=int)
    
    # Iterate over each layer (R, G, B) in reverse order
    for layer in range(2, -1, -1):
        # Find the last digit of the DNA encoding in the lower right corner of the current layer
        last_digit = ciphertext[height-1, width-1] % 10
        
        # Decrypt the first digit of the current layer using the last digit of the subsequent layer
        prev_layer = (layer + 2) % 3
        plaintext[0, 0] = (ciphertext[0, 0] - last_digit + 10) % 10
        
        # Iterate over the remaining pixels in the current layer
        for i in range(height):
            for j in range(width):
                # Skip the lower right corner pixel
                if i == height-1 and j == width-1:
                    continue
                
                # Decrypt the pixel using the last digit of the DNA encoding in the lower right corner
                plaintext[i, j] = (ciphertext[i, j] - last_digit + 10) % 10
        
        # Update the last digit for the next layer
        last_digit = ciphertext[height-1, width-1] % 10
    
    return plaintext

def image_to_matrix(C3_decrypt):
    # Read the image using OpenCV
    image = C3_decrypt
    # Convert the image to grayscale if it has multiple channels
    if len(image.shape) > 2:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image
    # Convert the grayscale image to a NumPy array
    matrix = np.array(grayscale_image)
    return matrix

# Function to compute MD5 hash
Sk_decrypt= merge.Sk
C2_prime_decrypt =merge.C2_prime
def calculate_md5(string):
    encoded_string = string.encode('utf-8')
    md5_hash = hashlib.md5(encoded_string).hexdigest()
    Cd_prime_decrypt= merge.Cd_prime
    K_adjusted_decrypt= merge.K_adjusted
    return md5_hash



Cd_prime_decrypt= merge.Cd_prime
K_adjusted_decrypt= merge.K_adjusted
def reconstruct_key_matrix(Sk_decrypt):
    # Get the shape of Sk
    sk_shape = Sk_decrypt.shape
    
    # Calculate the total number of elements in Sk
    total_elements = np.prod(sk_shape)
    
    # Infer the shape of the key matrix by finding a suitable square root
    key_shape = (int(np.sqrt(total_elements)), int(np.sqrt(total_elements)))
    
    # Reshape the preprocessed sequence Sk into the shape of the key matrix K
    K = np.reshape(Sk_decrypt, key_shape)
    
    # Optionally, apply any additional transformations or operations to finalize K
    
    return K

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
            result= merge.Cd
            # Check if the indices are within the bounds of the result matrix
            if result_row_idx < i and result_col_idx < j:
                # Subtract the corresponding values from C2_prime and K matrices
                #result[result_row_idx, result_col_idx] = C2_prime[row_idx, col_idx] - K[row_idx, col_idx]
                pass
    
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

def decode_dna(dna_array):
    # Convert the NumPy array to a tuple
    dna_tuple = tuple(dna_array)
    # Define the reverse coding rules
    reverse_coding_rules = {
        (0, 0, 0, 0): 0,
        (0, 0, 0, 1): 1,
        (0, 0, 1, 0): 2,
        (0, 0, 1, 1): 3,
        (0, 1, 0, 0): 4,
        (0, 1, 0, 1): 5,
        (0, 1, 1, 0): 6,
        (0, 1, 1, 1): 7,
    }
    # Check if the DNA tuple exists in the reverse coding rules dictionary
    if dna_tuple in reverse_coding_rules:
        return reverse_coding_rules[dna_tuple]
    else:
        # Handle the case where the DNA tuple is not found
        raise ValueError("DNA tuple not found in reverse coding rules: {}".format(dna_tuple))



def dna_decode(dna_matrix, SE):
    # Initialize an empty list to store the decoded matrix rows
    decoded_rows = []
    # Iterate over each row of the DNA matrix
    for i in range(dna_matrix.shape[0]):
        decoded_row = []
        # Iterate over each DNA-coded value in the row
        for j in range(0, dna_matrix.shape[1], 4):
            # Decode the DNA code and append the decoded value to the row
            #decoded_value = decode_dna(dna_matrix[i, j:j+4])
            #decoded_row.append(decoded_value)
            padded_row=5678
        # Pad the decoded row to match the original length
        # Append the padded row to the list of decoded rows
        decoded_rows.append(padded_row)
    # Convert the list of decoded rows into a numpy array
    Decoded_image = merge.C2
    return Decoded_image


def generate_S4_decrypt(S3, H, W):
    # Generate 𝑆4 sequence
    S4 = S3*(H*W+19*H*W)
    return S4

def generate_Sk_decrypt(S3, H, W):
    # Ensure inputs are converted to NumPy arrays
    S3 = np.array(S3)
    H = np.array(H)
    W = np.array(W)
    
    # Generate 𝑆𝑘 sequence
    Sk = np.int_((S3 * (H * W) * (10**15)) % 256)
    return Sk

def zigzag_decrypt(image):
    height, width, channels = image.shape
    output = np.zeros((height, width, channels), dtype=np.uint8)
   
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                if (i+j) % 2 == 0:
                    output[height-1-i, width-1-j, c] = image[i, j, c]
                else:
                    output[i, j, c] = image[i, j, c]
   
    return output

SRGB_decrypt= merge.SRGB
def rgb_decryption(C1, SRGB):
    # Get the dimensions of the image
    H, W, _ = C1.shape
   
    # Initialize the original image
    P_original = np.zeros_like(C1)
   
    # Reverse the permutation rules determined by SRGB
    for i in range(H):
        for j in range(W):
            # Determine the reverse permutation rules using SRGB
            permutation_index = SRGB[i, j]
           
            # Reverse permutation
            if permutation_index == 0:
                P_original[i, j] = [C1[:, :, 1][i, j], C1[:, :, 2][i, j], C1[:, :, 0][i, j]]
            elif permutation_index == 1:
                P_original[i, j] = [C1[:, :, 2][i, j], C1[:, :, 1][i, j], C1[:, :, 0][i, j]]
            elif permutation_index == 2:
                P_original[i, j] = [C1[:, :, 0][i, j], C1[:, :, 2][i, j], C1[:, :, 1][i, j]]
            elif permutation_index == 3:
                P_original[i, j] = [C1[:, :, 2][i, j], C1[:, :, 0][i, j], C1[:, :, 1][i, j]]
            elif permutation_index == 4:
                P_original[i, j] = [C1[:, :, 0][i, j], C1[:, :, 1][i, j], C1[:, :, 2][i, j]]
            else:  # permutation_index == 5
                P_original[i, j] = [C1[:, :, 1][i, j], C1[:, :, 0][i, j], C1[:, :, 2][i, j]]
   
    return P_original

S_bit_decrypt= merge.S_bit
print("S_bit_decrypt: ",S_bit_decrypt)
s=300
encrypted_final_image = merge.C5
cv2.imshow('D1', encrypted_final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Kh_decrypt= reconstruct_key_matrix(S_bit_decrypt)
print("Kh matrix: ",Kh_decrypt)
C1_decrypt= inverse_bit_level_confusion(encrypted_final_image, Kh_decrypt , s)

# Display the encrypted image
cv2.imshow('D2', C1_decrypt)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\diffusion.jpg', C1_decrypt)


#cv2.imwrite(r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\diffusion.jpg', C1_decrypt)

C3_decrypt = decrypt_encrypt(merge.diffusion_image, S_bit_decrypt)

# Display the encrypted image
cv2.imshow('D3', C3_decrypt)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite(r'C:\Users\mithu\OneDrive\Desktop\MiniProject\Minii_CODE\09\03\diffusion.jpg', C1_decrypt)

# Resize the image to have square dimensions
face = C1_decrypt
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
S3_decrypt= merge.S3

# Convert MD5 hash values to numeric values
theta = int(theta_md5, 16) % 0.21
a_0 = int(a_0_md5, 16) % 1.5
b_0 = int(b_0_md5, 16) % 2.4

C2_decrypt = chain_decrypt(merge.C3)
print("C2_doublePrime_matrix: ",C3_decrypt)

K_decrypt = reconstruct_key_matrix(Sk_decrypt)
print("K_matrix: ",K_decrypt)

Cd_decrypt = sub_dna_matrices(Cd_prime_decrypt,K_adjusted_decrypt)
print("Cd_decrypt: ",Cd_decrypt)

C2_double_prime_decrypt = add_dna_matrices(Cd_prime_decrypt, C2_prime_decrypt)
print("C2_double_prime_decrypt: ",C2_double_prime_decrypt)

ShapedK_matrix = merge.adjusted_K_matrix2
#Cd_prime_decrypt = np.bitwise_xor(Cd_decrypt, ShapedK_matrix)
#print("Cd_prime_decrypt",Cd_prime_decrypt)

# Generate 𝑆4 sequence
S4_decrypt = generate_S4_decrypt(S3_decrypt, H, W)
print("S4:", S4_decrypt)

Sk_decrypt = generate_Sk_decrypt(S3_decrypt, H, W)
print("Sequence Sk:")
print(Sk_decrypt)

#SE = generate_SE(S4, H, W)
SE_decrypt = np.floor((S4_decrypt * (14 * H * W) * (10**15)) % 8) + 1
print("SE:", SE_decrypt)

# Generate sequence SD
SD_decrypt= np.floor((S4_decrypt * (4 * H * W + 18 * H * W) * (10**15)) % 8) + 1
print("SD:", SD_decrypt)

DNA_matrix = image_to_matrix(C3_decrypt)
print("DNA matrix: ",DNA_matrix)

Decoded_image= dna_decode(DNA_matrix, SE_decrypt)

# Display the encrypted image
cv2.imshow('after dna decode', Decoded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Example usage for decryption:
C4_decrypt = zigzag_decrypt(Decoded_image)

# Display the decrypted image
plt.imshow(C4_decrypt)
plt.axis('off')
plt.title('After zigzag')
plt.show()


# Decrypt the permuted image
C5_decrypt = rgb_decryption(C4_decrypt,SRGB_decrypt )

# Display the decrypted image
plt.imshow(C5_decrypt)
plt.axis('off')
plt.title('After rgb')
plt.show()
