from PIL import Image
import numpy as np

def image_to_matrix(image_path, output_txt):
    # Open the image
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    
    # Convert to NumPy array
    matrix = np.array(img, dtype=int)
    height, width = matrix.shape
    
    # Save matrix to a text file with height and width as the first two values
    with open(output_txt, "w") as f:
        f.write(f"{height} {width}\n")
        np.savetxt(f, matrix, fmt="%d")
    
    return matrix

# Example usage
image_path = "image.jpg"  # Replace with the actual path
output_txt = "matrix_output.txt"  # Output file name
matrix = image_to_matrix(image_path, output_txt)
print(f"Matrix saved to {output_txt}")