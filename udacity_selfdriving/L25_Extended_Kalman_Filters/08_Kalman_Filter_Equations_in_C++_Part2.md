## Programming Quiz Solution

Below, you'll find a video with the solution and another code editor below with Andrei's solution for you to play with.

**see the video**

> *Note:* Certain small changes have been made between the video and the solution below in order to approve the readability of the code and follow more consistent coding style, but should have no impact on the actual processing and output of the code. For instance, we've only scoped in the functions from the `std` namespace we want to use - you can check out this [StackOverflow post](https://stackoverflow.com/questions/1452721/why-is-using-namespace-std-considered-bad-practice) for why you might want to avoid `using namespace std`.

```cpp
void filter(VectorXd &x, MatrixXd &P) {

  for (unsigned int n = 0; n < measurements.size(); ++n) {

    VectorXd z = measurements[n];
    // TODO: YOUR CODE HERE
    /**
     * KF Measurement update step
     */
    VectorXd y = z - H * x;
    MatrixXd Ht = H.transpose();
    MatrixXd S = H * P * Ht + R;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P * Ht * Si;

    // new state
    x = x + (K * y);
    P = (I - K * H) * P;

    /**
     * KF Prediction step
     */
    x = F * x + u;
    MatrixXd Ft = F.transpose();
    P = F * P * Ft + Q;

    cout << "x=" << endl <<  x << endl;
    cout << "P=" << endl <<  P << endl;
  }
}
```



## Quiz

Why do we not use the process noise in the state prediction function, even though the state transition equation has one? In other words, why does the code set `u << 0`, 0 for the equation $x = F * x + u$ ?

1. Because the process is just an approximation and we do not include the noise.
2. The noise mean is zero.
3. We should! There’s a bug in the function.
4. The noise mean is too large.

---

**Ans**: 2

> Looking closely at the process noise, we know from the Kalman Filter algorithm that its mean is zero and its covariance matrix is usually noted by $Q∗N(0,Q)​$. The first equation only predicts the mean state. As the mean value of the noise is zero, it does not directly affect the predicted state. However, we can see that the noise covariance $Q​$ is added here to the state covariance prediction so that the state uncertainty always increases through the process noise.

