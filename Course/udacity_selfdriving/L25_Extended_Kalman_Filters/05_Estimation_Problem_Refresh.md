**see the video**

## Definition of Variables

- xxx is the mean state vector. For an extended Kalman filter, the mean state vector contains information about the object's position and velocity that you are tracking. It is called the "mean" state vector because position and velocity are represented by a gaussian distribution with mean xxx.
- PPP is the state covariance matrix, which contains information about the uncertainty of the object's position and velocity. You can think of it as containing standard deviations.
- k represents time steps. So xkx_kxk refers to the object's position and velocity vector at time k.
- The notation k+1∣kk+1|kk+1∣k refers to the prediction step. At time k+1k+1k+1, you receive a sensor measurement. Before taking into account the sensor measurement to update your belief about the object's position and velocity, you predict where you think the object will be at time k+1k+1k+1. You can predict the position of the object at k+1k+1k+1 based on its position and velocity at time kkk. Hence xk+1∣kx_{k+1|k}xk+1∣k means that you have predicted where the object will be at k+1k+1k+1 but have not yet taken the sensor measurement into account. 
- xk+1x_{k+1}xk+1 means that you have now predicted where the object will be at time k+1k+1k+1 and then used the sensor measurement to update the object's position and velocity.

![img](../data/L25_5.png)

---

## Quiz

What should a Kalman Filter do if both the radar and laser measurements arrive at the same time, **k+3**?
> Hint: The Kalman filter algorithm predicts -> updates -> predicts  -> updates, etc. If two sensor measurements come in simultaneously,  the time step between the first measurement and the second measurement  would be zero.

1. Predict the state to k+3, and then update the state with only one of those measurements, either laser or radar.
2. Predict the state to k+3, and then only update with the laser measurement because it is more accurate
3. Predict the state to k+3 then use either one of the sensors to update. Then predict the state to k+3 again and update with the other sensor measurement.
4. Skip the prediction step beacause we have two measurements. Just update the prior probability distribution twice.

---

**Ans**: 3

As you saw, the Kalman filter is a two-step process: predict, and then update. If you receive two measurements simultaneously, you can use this process with either measurement and then repeat the process with the other measurement. The order does not matter!