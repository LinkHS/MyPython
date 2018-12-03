## Locate the Lane Lines

You now have a thresholded warped image and you're ready to map out the lane lines! There are many ways you could go about this, but here's one example of how you might do it:

### Line Finding Method: Peaks in a histogram
After applying calibration, thresholding, and a perspective transform to a road image, you should have a binary image where the lane lines stand out clearly. However, you still need to decide explicitly whcih pixels are part of the liens and which belong to the left line and which belong to the right line. 

Plotting a historgram of where the binary activations occur across the image is one potential solution for this. In the quiz below, let's take a couple quick steps to create our histogram!

```{.python .input  n=2}
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline 

# Load our image
# `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1
img = mpimg.imread('../data/L9_6_warped_example.jpg')/255

def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[int(img.shape[0]/2):]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

# Create histogram of image binary activations
histogram = hist(img)

# Visualize the resulting histogram
plt.plot(histogram)
```

Here's the approach I took.

I take a histogram along all the columns in the **lower half** of the image like this:
 ```python
import numpy as np
import matplotlib.pyplot as plt

histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
plt.plot(histogram)
 ```

---
## Sliding Window
With this histogram we are adding up the pixel values along each column in the image. In our thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the the x-position of the base of the lane lines. We can use that as a starting point for where to search for the lines. From that point, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

Here is a short animation (no sound!) showing this method:

```{.python .input}
from IPython.display import HTML
HTML("""
<iframe width="715" height="402" src="https://www.youtube.com/embed/siAMDK8C_x8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>```
""")
```
