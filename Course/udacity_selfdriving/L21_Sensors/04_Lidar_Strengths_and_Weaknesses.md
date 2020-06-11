**see the video**

|             | CAMERA | LIDAR  | RADAR |
| ----------- | ------ | ------ | ----- |
| RESOLUTION  | HIGH   | MEDIUM | LOW   |
| NOISE       | HIGH   | LOW    | LOW   |
| VELOCITY    | LOW    | LOW    | HIGH  |
| ALL-WEATHER | LOW    | LOW    | HIGH  |
| SIZE        | HIGH   | LOW    | HIGH  |

> Higher == Better

---
**Footnote on Lidar**

There are other possibilities to scan the laser beams. Instead of rotating the lasers or having a rotating mirror, we can scan the lidar with a vibrating micromirror. Those lidars are in development but none are commercially available now (as of March 2017).

Instead of mechanically moving the laser beam, a similar principle to phased array radar can be employed. Dividing a single laser beam into multiple waveguides, the phase relationship between the waveguides can be altered and thereby the direction of the laser beam shifted. A company named [Quanergy](http://quanergy.com/) is working on systems like that. The advantage is that the form factor can be much smaller and that there are no moving parts.

Another possibility is to use the laser as a gigantic flash like with a camera and then measuring the arrival times for all the objects with one big imaging photodiode array. This is in effect a 3D camera. The components are currently very expensive and currently this is used more in space and in terrain mapping applications. 

