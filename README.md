# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:

Import necessary libraries such as OpenCV, NumPy, and Matplotlib for image processing and visualization.

### Step2:

Read the input image using cv2.imread() and store it in a variable for further processing.


### Step3:

Apply various transformations like translation, scaling, shearing, reflection, rotation, and cropping by defining corresponding functions:

1.Translation moves the image along the x or y-axis.
2.Scaling resizes the image by scaling factors.
3.Shearing distorts the image along one axis.
4.Reflection flips the image horizontally or vertically.
5.Rotation rotates the image by a given angle.

### Step4:
Display the transformed images using Matplotlib for visualization. Convert the BGR image to RGB format to ensure proper color representation.

### Step5:
Save or display the final transformed images for analysis and use plt.show() to display them inline in Jupyter or compatible environments.
<br>

## Program:
```python
Developed By: Jaisree B
Register Number: 212224230100


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display image using Matplotlib
def display_image(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper color display
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load an image
image = cv2.imread('image1.png')
display_image(image, 'Original Image')


# i) Image Translation
def translate(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return translated

translated_image = translate(image, 100, 50)
display_image(translated_image, 'Translated Image')

# ii) Image Scaling
def scale(img, scale_x, scale_y):
    scaled = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return scaled

scaled_image = scale(image, 1.5, 1.5)
display_image(scaled_image, 'Scaled Image')

# iii) Image Shearing
def shear(img, shear_factor):
    rows, cols, _ = img.shape
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(img, M, (cols, rows))
    return sheared

sheared_image = shear(image, 0.5)
display_image(sheared_image, 'Sheared Image')

# iv) Image Reflection
def reflect(img):
    reflected = cv2.flip(img, 1)  # 1 for horizontal flip
    return reflected

reflected_image = reflect(image)
display_image(reflected_image, 'Reflected Image')

# v) Image Rotation
def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

rotated_image = rotate(image, 45)
display_image(rotated_image, 'Rotated Image')

# vi) Image Cropping
def crop(img, start_row, start_col, end_row, end_col):
    cropped = img[start_row:end_row, start_col:end_col]
    return cropped

cropped_image = crop(image, 50, 50, 200, 200)
display_image(cropped_image, 'Cropped Image')




```
## Output:

<img width="595" height="532" alt="Screenshot 2026-02-05 142313" src="https://github.com/user-attachments/assets/68e08303-6acd-4fd7-ab19-0ea51481e4e8" />

### i)Image Translation
<br>
<img width="424" height="534" alt="Screenshot 2026-02-05 142325" src="https://github.com/user-attachments/assets/7e57f33c-953f-404c-8ea5-750545fd535a" />

<br>

### ii) Image Scaling
<br>
<img width="439" height="521" alt="Screenshot 2026-02-05 142338" src="https://github.com/user-attachments/assets/badc914a-cfb1-40b1-a1b6-ccf5e6ac5e93" />

<br>


### iii)Image shearing
<br>
<img width="435" height="520" alt="Screenshot 2026-02-05 142347" src="https://github.com/user-attachments/assets/fe5e2339-ea44-484e-a418-b7e7027c51d0" />

<br>


### iv)Image Reflection
<br>
<img width="455" height="517" alt="Screenshot 2026-02-05 142356" src="https://github.com/user-attachments/assets/80fd4faa-aa69-44bf-8ffe-a8eaae043d00" />

<br>



### v)Image Rotation
<br>
<img width="462" height="518" alt="Screenshot 2026-02-05 142405" src="https://github.com/user-attachments/assets/b3675b7a-3d58-4839-a85c-033a10e3493a" />

<br>



### vi)Image Cropping
<br>
<img width="516" height="530" alt="Screenshot 2026-02-05 142413" src="https://github.com/user-attachments/assets/53ed1999-a9c9-4989-8d28-ee42ad563832" />

<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
