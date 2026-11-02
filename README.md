# Week 3 Deliverable: Classical Vehicle Motion Tracker

## Overview
This repository contains a classical Computer Vision implementation for tracking vehicles across video frames using sparse feature extraction and geometric displacement techniques.

## Features Implemented
* **Sparse Optical Flow (Lucas-Kanade):** Tracks displacement vectors of Shi-Tomasi corners across consecutive frame pyramids.
* **Feature Descriptor Matching (ORB):** Leverages binary descriptors with Hamming distance and Lowe's Ratio Test for geometric verification.
* **Trajectory Overlay & Motion Visualizer:** Dynamic path generation and real-time motion vector rendering.

## Visual Proof & Visual Demonstration
Below is the animated visual verification showing feature point trajectories and vehicle motion tracking in action:

![Vehicle Motion Tracker Proof](assets/vehicle_tracking_demo.gif)

## How to Run
```bash
# 1. Install Dependencies
pip install opencv-python imageio numpy

# 2. Run Tracker Script
python vehicle_tracker.py
