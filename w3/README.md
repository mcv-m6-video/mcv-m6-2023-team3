# Week 3

During the third week of our project, we focused on object detection and object tracking. We worked on off-the-shelf object detection models, annotation of data for fine-tuning, and fine-tuning models for specific data. We also used different ways of doing K-Fold Cross-validation to evaluate the performance of the models. Additionally, we explored object tracking techniques such as tracking by overlap and Kalman Filter. To evaluate the performance of the tracking methods, we used IDF1 and HOTA scores. Our goal was to explore various techniques to detect and track objects in images and evaluate their effectiveness.


## Available tasks

* **Task 1**: Object detection
  * **Task 1.1**: Off-the-shelf
  * **Task 1.2**: Annotation
  * **Task 1.3**: Fine-tune to your data
  * **Task 1.4**: K-Fold Cross-validation
* **Task 2**: Object tracking
  * **Task 2.1**: Tracking by overlap
  * **Task 2.2**: Tracking with a Kalman Filter
  * **Task 2.3**: IDF1,HOTA scores


## Usage
In order to run this project you need to execute the main.py from [the main page](https://github.com/mcv-m6-video/mcv-m6-2023-team3).
  ```
python main.py -h
usage: main.py [-h] <-w WEEK> <-t TASK>

M6 - Video Analysis: Video Surveillance for Road Traffic Monitoring

Arguments:
 -h, --help            show this help message and exit
 -w WEEK, --week WEEK  week to execute. Options are [1,2,3,4,5]
 -t TASK, --task TASK  task to execute. Options depend on each week.
  ```

## Slides

The slides for w3 are available [here](https://docs.google.com/presentation/d/1bTaPiW5-V4t5nyi4mDJ3oiAD_aPqXAO3lbOn-PdhqBw/edit).

## Requirements

In order to run the code you will need to clone the repository https://github.com/liruoteng/OpticalFlowToolkit.git to read the optical flow from the Kitty dataset 
