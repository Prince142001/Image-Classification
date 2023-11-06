import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('test.png',0)
_, th1 = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
_, th2 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
_, th3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
_, th4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
_, th5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, th1 ,th2 ,th3 ,th4, th5]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

#code added by Coding-With-Shivam 



# first code
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def display_images(images, titles, rows, cols, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    for i in range(len(images)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply different filters
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(img, -1, kernel)
        blur = cv2.blur(img, (5, 5))
        gblur = cv2.GaussianBlur(img, (5, 5), 0)
        median = cv2.medianBlur(img, 5)
        bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)

        filter_titles = ['2D Convolution', 'Blur', 'Gaussian Blur', 'Median', 'Bilateral Filter']
        filter_images = [dst, blur, gblur, median, bilateralFilter]

        # Apply edge detection
        lap = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)
        lap = np.uint8(np.absolute(lap))
        sobelX = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobelCombined = cv2.bitwise_or(sobelX, sobelY)

        edge_titles = ['Laplacian', 'Sobel X', 'Sobel Y', 'Sobel Combined']
        edge_images = [lap, sobelX, sobelY, sobelCombined]

        # Additional features: Denoising, Sharpening, Shape Recognition (Contour Analysis)
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        sharpened = cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

        gray_sharpened = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray_sharpened, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        shapes_detected = len(contours)

        additional_titles = ['Denoised Image', 'Sharpened Image', f'Shapes Detected: {shapes_detected}']
        additional_images = [denoised_img, sharpened, gray_img]

        # Display all the processed images
        all_images = [img] + filter_images + edge_images + additional_images
        all_titles = ['Original Image'] + filter_titles + edge_titles + additional_titles
        display_images(all_images, all_titles, 5, 3, figsize=(15, 15))

# Create the Tkinter GUI
root = tk.Tk()
root.title("Image Processing App")

canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()

open_button = tk.Button(root, text="Open Image", command=open_file, padx=20, pady=10, fg="white", bg="blue")
open_button.pack()

root.mainloop()

# second code
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def display_images(images, titles, rows, cols, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    for i in range(len(images)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply different filters
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(img, -1, kernel)
        blur = cv2.blur(img, (5, 5))
        gblur = cv2.GaussianBlur(img, (5, 5), 0)
        median = cv2.medianBlur(img, 5)
        bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)

        filter_titles = ['2D Convolution', 'Blur', 'Gaussian Blur', 'Median', 'Bilateral Filter']
        filter_images = [dst, blur, gblur, median, bilateralFilter]

        # Apply edge detection
        lap = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=3)
        lap = np.uint8(np.absolute(lap))
        sobelX = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobelCombined = cv2.bitwise_or(sobelX, sobelY)

        edge_titles = ['Laplacian', 'Sobel X', 'Sobel Y', 'Sobel Combined']
        edge_images = [lap, sobelX, sobelY, sobelCombined]

        # Additional functionalities: Resize, Rotate, Flip
        resized_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        flipped_img = cv2.flip(img, 1)  # Flip horizontally

        additional_titles = ['Resized', 'Rotated', 'Flipped']
        additional_images = [resized_img, rotated_img, flipped_img]

        # Compute and display histogram of grayscale image
        plt.hist(gray_img.ravel(), 256, [0, 256])
        plt.title('Histogram')
        plt.show()

        # Display all the processed images including additional functionalities
        all_images = [img] + filter_images + edge_images + additional_images
        all_titles = ['Original Image'] + filter_titles + edge_titles + additional_titles
        display_images(all_images, all_titles, 4, 5, figsize=(15, 12))

# Create the Tkinter GUI
root = tk.Tk()
root.title("Image Processing App")

open_button = tk.Button(root, text="Open Image", command=open_file)
open_button.pack()

root.mainloop()

