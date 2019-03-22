## Kalman Filter Equations in C++

Now, let's do a quick refresher of the Kalman Filter for a simple 1D motion case. Let's say that your goal is to track a pedestrian with state xxx that is described by a position and velocity. 

$$x = \begin{pmatrix} p \\ v \end{pmatrix}​$$



### Prediction Step

When designing the Kalman filter, we have to define the two linear functions: the state transition function and the measurement function. The state transition function is

$x′=F∗x+noise$,

where,

$$F = \begin{pmatrix} 1 & \Delta t \\ 0 & 1 \end{pmatrix}​$$

and $x′​$ is where we predict the object to be after time $\Delta t​$. 

$F$ is a matrix that, when multiplied with $x$, predicts where the object will be after time $\Delta t$.

By using the linear motion model with a constant velocity, the new location, $p′$ is calculated as 

$p' = p + v * \Delta t$,

where $p$ is the old location and $v$, the velocity, will be the same as the new velocity ($v′=v$) because the velocity is constant. 

We can express this in a matrix form as follows:

$$\begin{pmatrix} p' \\ v' \end{pmatrix} = \begin{pmatrix}1 & \Delta t \\ 0 & 1 \end{pmatrix} \begin{pmatrix} p \\ v \end{pmatrix}​$$

Remember we are representing the object location and velocity as gaussian distributions with mean $x$. When working with the equation $x′=F∗x+noise$, we are calculating the mean value of the state vector. The noise is also represented by a gaussian distribution but with mean zero; hence, noise = 0 is saying that the mean noise is zero. The equation then becomes $x′=F∗x$

But the noise does have uncertainty. The uncertainty shows up in the $Q$ matrix as acceleration noise.



### Update Step

For the update step, we use the measurement function to map the state vector into the measurement space of the sensor. To give a concrete example, lidar only measures an object's position. But the extended Kalman filter models an object's position and velocity. So multiplying by the measurement function H matrix will drop the velocity information from the state vector xxx. Then the lidar measurement position and our belief about the object's position can be compared.

$z=H∗x+w$

where $w$ represents sensor measurement noise.

So for lidar, the measurement function looks like this:

$z=p′​$.

It also can be represented in a matrix form:

$$z=\begin{pmatrix} 1 & 0 \end{pmatrix} \begin{pmatrix} p' \\ v' \end{pmatrix}$$.

As we already know, the general algorithm is composed of a prediction step where I predict the new state and covariance, $P$. 

And we also have a measurement update (or also called many times a correction step) where we use the latest measurements to update our estimate and our uncertainty.

**see the video**

Here is a download link to the [Eigen Library](https://d17h27t6h515a5.cloudfront.net/topher/2017/March/58b7604e_eigen/eigen.zip) that is being used throughout the programming assignments. Further details regarding this library can be found [here](http://eigen.tuxfamily.org/).

Note: In the classroom editor we are calling just Dense instead of Eigen/Dense as seen in videos. This is because the Eigen library had to have its folder structure reformatted to work with the programming quiz editor. If you run the code on your own computer you would still use Eigen/Dense.



**Notes for using the Eigen Library:**

You can create a vertical vector of two elements with a command like this:

```cpp
VectorXd my_vector(2);
```

You can use the so called comma initializer to set all the coefficients to some values:

```cpp
my_vector << 10, 20;
```

and you can use the cout command to print out the vector:

```cpp
cout << my_vector << endl;
```

The matrices can be created in the same way. For example, This is an initialization of a 2 by 2 matrix with the values 1, 2, 3, and 4:

```cpp
MatrixXd my_matrix(2,2);
my_matrix << 1, 2,
             3, 4;
```

You can use the same comma initializer or you can set each matrix value explicitly. For example, that's how we can change the matrix elements in the second row:

```cpp
my_matrix(1,0) = 11;  //second row, first column
my_matrix(1,1) = 12;  //second row, second column
```

Also, you can compute the transpose of a matrix with the following command:

```cpp
MatrixXd my_matrix_t = my_matrix.transpose();
```

And here is how you can get the matrix inverse:

```cpp
MatrixXd my_matrix_i = my_matrix.inverse();
```

For multiplying the matrix m with the vector b you can write this in one line as let’s say matrix c equals m times v:

```cpp
MatrixXd another_matrix;
another_matrix = my_matrix*my_vector;
```


### Programming Assignment TODOs 

Note that in the quiz below, in the `filter()` function, we actually do the measurement and then the prediction in the loop. Over time, the order of these doesn't have a huge impact, since it is just a cycle from one to the other. Here, the first thing you need is a measurement because otherwise there is no location information or even information that the object exists unless a sensor picked it up. So, you initialize location values with the measurement.

Task List

- Implement the prediction step within the loop.
- Implement the measurement update step within the loop.

```cpp
/** 
 * Write a function 'filter()' that implements a multi-
 *   dimensional Kalman Filter for the example given
 */

#include <iostream>
#include <vector>
#include "Dense"

using std::cout;
using std::endl;
using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;

// Kalman Filter variables
VectorXd x;	// object state
MatrixXd P;	// object covariance matrix
VectorXd u;	// external motion
MatrixXd F; // state transition matrix
MatrixXd H;	// measurement matrix
MatrixXd R;	// measurement covariance matrix
MatrixXd I; // Identity matrix
MatrixXd Q;	// process covariance matrix

vector<VectorXd> measurements;
void filter(VectorXd &x, MatrixXd &P);


int main() {
  /**
   * Code used as example to work with Eigen matrices
   */
  // design the KF with 1D motion
  x = VectorXd(2);
  x << 0, 0;

  P = MatrixXd(2, 2);
  P << 1000, 0, 0, 1000;

  u = VectorXd(2);
  u << 0, 0;

  F = MatrixXd(2, 2);
  F << 1, 1, 0, 1;

  H = MatrixXd(1, 2);
  H << 1, 0;

  R = MatrixXd(1, 1);
  R << 1;

  I = MatrixXd::Identity(2, 2);

  Q = MatrixXd(2, 2);
  Q << 0, 0, 0, 0;

  // create a list of measurements
  VectorXd single_meas(1);
  single_meas << 1;
  measurements.push_back(single_meas);
  single_meas << 2;
  measurements.push_back(single_meas);
  single_meas << 3;
  measurements.push_back(single_meas);

  // call Kalman filter algorithm
  filter(x, P);

  return 0;
}


void filter(VectorXd &x, MatrixXd &P) {

  for (unsigned int n = 0; n < measurements.size(); ++n) {
    // TODO: YOUR CODE HERE
		
    // KF Measurement update step

    // new state

    // KF Prediction step
        
    cout << "x=" << endl <<  x << endl;
    cout << "P=" << endl <<  P << endl;
  }
}
```

