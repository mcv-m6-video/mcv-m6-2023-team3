## Week 5

During this week we are goint to follow with the Multi-object Single camara tracking and the additonal of Multi-object Multi-camara tracking challenge.
The proble will be faced on the [AICity Challenge 2020](https://www.aicitychallenge.org/): **City-Scale Multi-Camera Vehicle Tracking.**
The goal is to track vehicles across multiple cameras arround a city. 

We have proposed the following solution for each task
* **Multi-target single-camera (MTSC) tracking**
    * Tracking from detections in  Maske R-CNN, SSD512 and Yolo3 backbones.
    * Create a solution based on tracking by overlap and Kalman tracker.
    * Remove static cars based on the distance traveled along the video

* **Multi-target multi-camera (MTMC) tracking**
   * Tracking from detections in  Maske R-CNN, SSD512 and Yolo3 backbones.
   * Train [NCA](http://contrib.scikit-learn.org/metric-learn/generated/metric_learn.NCA.html) model with Hue Histogram features to compute embedding for each track
   * Compare each track of one camera to each one of other cameras and decide a match based on a threshold.



## Slides

The Final slides are [here](https://docs.google.com/presentation/d/1COxV1K5cBSR6HK9wBdCDpgl8XroO2eR8A5PmY2dSccE/edit#slide=id.g23aac86a548_0_33).

You can also visualize the week 5/final presentation in a [static version](https://github.com/mcv-m6-video/mcv-m6-2023-team3/blob/main/Final%20presentation.pdf).

## Report
The final report is available [here](https://github.com/mcv-m6-video/mcv-m6-2023-team3/raw/main/M6_paper.pdf)
