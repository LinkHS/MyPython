# Lane_Curvature

watch the video

---
## Calculating Lane Curvature

Self-driving cars need to be told the correct steering angle to turn, left or right. You can calculate this angle if you know a few things about the speed and dynamics of the car and how much the lane is curving.

One way to calculate the curvature of a lane line, is to fit a 2nd degree polynomial to that line, and from this you can easily extract useful information.

For a lane line that is close to vertical, you can fit a line using this formula: $f(y) = Ay^2 + By + C$, where $A$, $B$, and $C$ are coefficients.

$A$ gives you the curvature of the lane line, $B$ gives you the heading or direction that the line is pointing, and $C$ gives you the position of the line based on how far away it is from the very left of an image (y = 0).

