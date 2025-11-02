# ASTRA - Autonomous System for Tracking, Reconnaissance, and Assistance
An open-source UAV platform for search-and-rescue (SAR) that combines real-time human detection, waypoint navigation, and ground-station visualization

![1744755671148](https://github.com/user-attachments/assets/ddf777fc-4f2d-4563-b751-98230e194a4a)

## Table of contents
- [What is ASTRA](#what-is-astra?)
- [Core Capabilities](#core-capabilities) 
- [System Architecture (High Level)](#system-architecture-(high-level))
- [Key specs & design targets](#)
- [Demos](#heading-title)
- [Repository layout](#heading-title)
- [Getting started](#heading-title)
- [How it works (data flow)](#heading-title)
- [Hardware bill of materials](#heading-title)
- [Testing highlights](#heading-title)
- [Roadmap](#heading-title)
- [Safety notes](#heading-title)
- [Team](#heading-title)
- [Acknowledgements](#heading-title)


## What is ASTRA?
ASTRA is a semi-autonomous quadrotor designed to improve SAR operations. It flies in versatile environments, detect humans on-board using AI, and relays detections + GPS coordinates to a ground station UI for immediate response. ASTRA is the better alternative as it is a targeted, low-cost research platform that bridges manual piloting and intelligent rescue tooling.

## Core Capabilities 
- '''On-board computer vision (YOLOv8):''' Real-time human detection (single class: person) driven by NVIDIA Jetson Orin Nano + Luxonis OAK-D-Lite depth camera.
- '''Waypoint/simulated autonomy:''' Routes planned in QGroundControl (Pixhawk 4/PX4 Firmware).
- '''Live communication:''' Detections and telemetry to the ground station over UDP/Wi-Fi; MavLink telemetry maintained over RF
- '''Ground-station UI:''' Camera feed, Google Maps GPS view (online) with Offline Mode feedback map when there's no internet

## System Architecture (High Level)
ASTRA is split into UAV and Ground station subsystems:
- **UAV:** Pixhawk 4 (flight), M10 GPS, OAK-D-LITE
- **Ground Station:** Python/Tkinter UI showing detections, map, and terminal telemetry; live stream via Flask
### Summary
GPS --> Jetson (Processing + CV) --> UDP --> UI
RF telemetry is maintained in parallel
I2C/USB/UART form the internal peripheral backbone, and UDP carries time-sensitive events

