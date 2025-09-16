# Object Detection using YOLO

In this final blog of the series of projects, we will be shifting focus from OpenCV to the current standard algorithm in real-time object detection, namely, YOLO, or You-Only-Look-Once.

### Imports

`!pip install ultralytics`

```
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

from ultralytics import YOLO
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

## Training and Detecting our Custom Class

Where in the previous blogs we went over OpenCV's techniques for facial detection particularly, using YOLO, we will now be able detect anything we would want to, provided we have the necessary dataset to train the model on. In this case, we will be utilizing a custom dataset containing all the necessary training, testing, and validation images in the proper format for YOLOv11.

If you want to make your own dataset, I would recommend utilizing the free features available to you on Roboflow. [This video](https://www.youtube.com/watch?v=a3SBRtILjPI&t=223s) is a huge help for determining how to go through the annotation process. For sourcing images for your dataset, [Kaggle](https://www.kaggle.com/datasets?tags=13207-Computer+Vision) is a great help! Make sure you cite your sources if using your dataset in a work you plan on publishing.

## Use Cases

YOLO is used even more frequently than the previous techniques we reviewed. It has become the standard in real-time object detection, thanks to its incredible speed, which is continually increasing as it is improved year after year. Be it faces, or cars, or any thing else that a user may want to detect, YOLO presents the tools for doing such things, and doing them quickly and efficiently, provided the user can supply it with a properly structured dataset.

## Training the Model

Compared to the amount of pre-processing and extra steps that were required when utilizing OpenCV, another advantage of YOLO becomes abundantly clear during implementation, namely how simple the entire pipeline is, from training to prediction. In the following code, we provide YOLO with out dataset, and it handles all the training. Because we are running 50 training epochs, it will run for roughly 9-12 minutes thanks to the GPU we are utilizing for this runtime. After that point, we will be able to move on.

**It is very important to note before moving forward that if you plan on training on your local machine, be aware that without a compatible graphics card, CUDA, and PyTorch installed, the training process will default to using your CPU, which will take a number of hours to complete, as opposed to taking under thirty minutes using a GPU. Proceed with caution if you have not installed these things and verified that PyTorch is detecting your GPU.**

```
# If you are running locally and want to verify your PyTorch and CUDA installation, run this cell
# If this returns "True", you are good to go
# If this returns "False", verify your installation of PyTorch and CUDA

import torch
print(torch.cuda.is_available())
```

```
# Load a model
model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

# Train the model
training = model.train(data="/content/fish_dataset.v4-fish-v4.yolov11/data.yaml", epochs=50, imgsz=640)
```

## Viewing Training Results

After running the training process, a number of different things are saved, presenting us with how the training process went. The most important of these is stored within the results.png image. Provided the training is going according to plan, what we should observe is an inverse relationship between the different types of loss and minimum average precision and recall.

My result is saved after training, but ensure you run this code after doing the training process to review your own results before continuing. Additionally, the path to the results image may change after training on your end, so verify in the previous code block where your results are saved to.

```
image = plt.imread("/content/runs/detect/train2/results.png")

# Display the results image
create_mpl_figure(10,10, [image], ["Results"])
```

<div style="display: flex; justify-content: space-around;">
    <div>
        <img src="../../images/yolo_custom_class/results.png" alt="Graphs of results during training">
    </div>
</div>

## Predicting the Custom Class on Test Images

Now that the model has been trained, and the results of its training have been observed to be reliable, we can now going about actually testing its ability to detect the class on which it was trained, which is in this case, fish.

In similar fashion to the displaying of the results image, ensure that the path to the image is correct after running the prediction. You may need to change the path for the pred_test variable according to wherever the results are saved.

```
# Run the model against a test image, verifying that it properly identifies the custom class
prediction = model("/content/fish_dataset.v4-fish-v4.yolov11/test/images/P1ROZC-Z_7_jpg.rf.745f9956e192bf19ef2ebd8a7ede9d26.jpg", save=True)
```

```
pred_test = plt.imread("/content/runs/detect/train23/P1ROZC-Z_7_jpg.rf.745f9956e192bf19ef2ebd8a7ede9d26.jpg")

# Display the test image after it is fed through the Model
create_mpl_figure(5,5, [pred_test], ["Prediction Test"])
```

<div style="display: flex; justify-content: space-around;">
    <div>
        <img src="../../images/yolo_custom_class/prediction.png">
    </div>
</div>

## Conclusion

And just like that, you are able to detect custom objects using the YOLO algorithm! Not only is it exceedingly fast, but it is very easy to implement and train custom models on!
