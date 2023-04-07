# Week 4
During the fourth week of our project, we focused on optical flow estimation and multi-target single-camera tracking. We explored different techniques for estimating optical flow, such as block matching and off-the-shelf methods. We also worked on improving tracking using optical flow. For multi-target single-camera tracking, we explored tracking by overlap and tracking with a Kalman filter. We evaluated the performance of these techniques using IDF1 and HOTA scores. Our goal was to explore various techniques to estimate optical flow and track multiple objects in a single camera image and evaluate their effectiveness.

## Available tasks


* **Task 1**: Optical flow 
  * **Task 1.1**: Optical flow estimation with block matching
  * **Task 1.2**: Optical flow estimation with off-the-shelf method
  * **Task 1.3**: Improve tracking with optical flow 
* **Task 2**: Multi-Target Single-Camera tracking
* **Task 3** (optional): Extend test sequences



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

The slides of task 1 are [here](https://docs.google.com/presentation/d/1FTtwSulFm87SZkPYsDEbVqKK0ixPBFlQ0KzYLtEYOio/edit#slide=id.p).

The slides of tasks 2 and 3 are [here](https://docs.google.com/presentation/d/1i7jyIbeC1bf1TXjsiLCqS8t2q83O2DqegVrfY0NBrIU/edit#slide=id.p).

## Requirements

In order to run the w4 code you will need: 

[Pyflow](https://github.com/pathak22/pyflow.git)

[Optical Flow Prediction](https://github.com/philferriere/tfoptflow)
