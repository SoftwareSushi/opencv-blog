# Blog Title

Description & introduction of image manipulation techniques

## Use Cases

List of use cases for related techniques of this part of the blog.

## Techniques

Explain each technique for this section of the blog (drawing on images, rotating images, etc.). Include code snippets. Show images and the effects of each technique when used.

Example:

### Drawing on Images:

**What it does:** Adds visual elements (shapes and text) to an image

**Why it matters:** Useful for annotation, visualization, or debugging during processing

```
import cv2

image = cv2.imread("input.jpg")
cv2.line(image, (50, 50), (200, 50), (0, 255, 0), 2)
cv2.rectangle(image, (50, 100), (200, 200), (255, 0, 0), 3)
cv2.circle(image, (150, 300), 40, (0, 0, 255), -1)
cv2.putText(image, "OpenCV", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imwrite("output_drawn.jpg", image)
```

**_Display the before and after on the image that has been edited_**
