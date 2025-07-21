# Face Recognition with OpenCV

Building on our previous blog wherein we covered the topic of face detection, we will now be using OpenCV to illustrate how not only can you use computers to detect faces, so too can you use them to recognize specific faces.

### Imports & Sample Images

```
!pip uninstall opencv-python opencv-contrib-python -y
%pip install opencv-contrib-python==4.10.0.84 --force-reinstall
```

```
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import base64
import os

from IPython.display import HTML, display
from io import BytesIO
from PIL import Image
```

### Utils

```
# Function for the creation of flexible MatPlotLib figures
def create_mpl_figure(w,h,images,titles="Image",axis="off",color_maps=None):
    plt.figure(figsize=[w,h])

    for i, image in enumerate(images):
        plt.subplot(1,len(images),i+1);

        if color_maps is None:
            plt.imshow(image);
        elif len(color_maps) > 1:
            plt.imshow(image, cmap=f"{color_maps[i]}");
        else:
            plt.imshow(image, cmap=f"{color_maps[0]}")

        plt.title(titles[i]);
        plt.axis(axis);
```

## Face Recognition Process

Unlike face detection, which was demonstrated in the previous blog, facial recognition is quite a bit more complex than using a pre-built method in OpenCV to process images. This blog will illustrate the use of the Local Binary Pattern Histogram (LBPH) method. This process functions by first taking as an input a face then dividing that face into a grid of small regions. From there, binary patterns are extracted by comparing each pixel within the small region to its neighboring pixels. Histograms are then created for each region, and finally, the histograms for every single region is concatenated to a vector representative of the particular face.

## Use Cases

Face Recognition is used all the time, whether it be for biometric authentication, use in Law Enforcement scenarios through CCTV, or a variety of other circumstances. The uses for it are growing consistently, and show no sign of slowing down.

## Utilizing Local Binary Pattern Histograms

### Fetching our sample faces

Unlike the previous blog, where we detected faces within a given image, we will be using an existing dataset of 400 images of 40 different faces at 10 images per face. If one were to instead want to do this with their own dataset, all of the same steps can be followed, just substituting the sample data with their own. For this purpose then, it is unnecessary to use Haar Cascades to first detect the faces, since the images are all cropped faces.

```
import urllib.request
import zipfile
import os

# Download the face dataset
url = "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"
zip_filename = "att_faces.zip"

print("Downloading face dataset...")
urllib.request.urlretrieve(url, zip_filename)
print("Download complete!")

# Create faces directory
os.makedirs("faces", exist_ok=True)

# Extract the zip file
print("Extracting files...")
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall("faces")
print("Extraction complete!")

# Clean up the zip file
os.remove(zip_filename)
print("Dataset ready for use!")
```

### Preparing the Data

After retrieving our face images, we now need to transform them into a usable format for the LBPH process. If you look closely, you will notice that of our sample size of 10 photos per person, 9 are fed to the recognizer, while 1 is held back, which will allow us to properly test later on whether or not the recognizer is able to accurately determine whether or not it recognizes a face.

```
base_path = './faces'
test_faces = []
test_labels = []
faces = []
labels = []

for label in range(1, 41):  # 40 people
    for i in range(1, 11):  # 10 images per person
        img_path = f'{base_path}/s{label}/{i}.pgm'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if i == 10:
            # Reserve the 10th image of each person for testing
            test_faces.append(img)
            test_labels.append(label)
        else:
            faces.append(img)
            labels.append(label)
```

### Training Using the Faces

Now that we have our faces downloaded, we need to train the recognizer to actually be able to see and recognize each of the 10 different faces. In order to do this, we needed to install a wrapper package called opencv-contrib-python. This library contains the cv2.face property, which we need to access for the training of the recognizer.

```
import cv2.face

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
```

### Testing the Recognizer

Now that we have fed the recognizer 9 pictures for each face, we will take the test faces and display them side by side to see how the recognizer does in ascertaining whether it recognizes a given face. It is very important to note here that confidence scoring in the LBPH method is an inverse to what one might normally assume.

Higher numbers = Lower confidence (100<=)
Lower numbers = Higher confidence (<=50)

```
trained_path_1 = 'faces/s1/1.pgm'
test_path_2 = 'faces/s1/10.pgm'

trained_path_3 = 'faces/s25/6.pgm'
test_path_4 = 'faces/s25/10.pgm'

trained_path_5 = 'faces/s39/2.pgm'
test_path_6 = 'faces/s39/10.pgm'

trained_img_1 = cv2.imread(trained_path_1, cv2.IMREAD_GRAYSCALE)
test_img_2 = cv2.imread(test_path_2, cv2.IMREAD_GRAYSCALE)

trained_img_3 = cv2.imread(trained_path_3, cv2.IMREAD_GRAYSCALE)
test_img_4 = cv2.imread(test_path_4, cv2.IMREAD_GRAYSCALE)

trained_img_5 = cv2.imread(trained_path_5, cv2.IMREAD_GRAYSCALE)
test_img_6 = cv2.imread(test_path_6, cv2.IMREAD_GRAYSCALE)

predicted_label_1, confidence_1 = recognizer.predict(trained_img_1)

predicted_label_2, confidence_2 = recognizer.predict(test_img_2)

predicted_label_3, confidence_3 = recognizer.predict(trained_img_3)

predicted_label_4, confidence_4 = recognizer.predict(test_img_4)

predicted_label_5, confidence_5 = recognizer.predict(trained_img_5)

predicted_label_6, confidence_6 = recognizer.predict(test_img_6)

imgs = [trained_img_1, test_img_2, trained_img_3, test_img_4, trained_img_5, test_img_6]

img_titles = [f'Trained\nPredicted: s{predicted_label_1}\nConfidence: {confidence_1:.2f}', f'Test\nPredicted: s{predicted_label_2}\nConfidence: {confidence_2:.2f}', f'Trained\nPredicted: s{predicted_label_3}\nConfidence: {confidence_3:.2f}', f'Test\nPredicted: s{predicted_label_4}\nConfidence: {confidence_4:.2f}', f'Trained\nPredicted: s{predicted_label_5}\nConfidence: {confidence_5:.2f}', f'Test\nPredicted: s{predicted_label_6}\nConfidence: {confidence_6:.2f}']

create_mpl_figure(15, 10, imgs, img_titles, color_maps=["gray"])
```

<div style="display: flex; justify-content: space-around;">
    <div>
        <img src="../../images/opencv_face_detection/face_recognition.png" alt="Images of Trained and Tested Images for Recognition">
    </div>
</div>

## Conclusion

That is how you implement facial recognition using OpenCV! Though a bit more complex than face detection, once the rudiments of the process are understood, its flexibility and usefulness become readily apparent. In the next blog, we will be doing yet another complex facial detection task, namely, the application of angle-correct custom facial filters, such as the overlaying of a dog's face onto a face captured within the image.
