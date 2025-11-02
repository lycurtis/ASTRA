# ASTRA - Autonomous System for Tracking, Reconnaissance, and Assistance
An open-source UAV platform for search-and-rescue (SAR) that combines real-time human detection, waypoint navigation, and ground-station visualization

![1744755671148](https://github.com/user-attachments/assets/ddf777fc-4f2d-4563-b751-98230e194a4a)

## Table of contents
- [What is ASTRA](#what-is-astra)
- [Core Capabilities](#core-capabilities) 
- [System Architecture (High Level)](#system-architecture-high-level)
- [Key Specs & Design Targets](#key-specs-#&-design-targets)
- [Demos](#demos)
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
- `On-board computer vision (YOLOv8):` Real-time human detection (single class: person) driven by NVIDIA Jetson Orin Nano + Luxonis OAK-D-Lite depth camera.
- `Waypoint/simulated autonomy:` Routes planned in QGroundControl (Pixhawk 4/PX4 Firmware).
- `Live communication:` Detections and telemetry to the ground station over UDP/Wi-Fi; MavLink telemetry maintained over RF
- `Ground-station UI:` Camera feed, Google Maps GPS view (online) with Offline Mode feedback map when there's no internet

## System Architecture (High Level)
ASTRA is split into UAV and Ground station subsystems:
- **UAV:** Pixhawk 4 (flight), M10 GPS, OAK-D-LITE
- **Ground Station:** Python/Tkinter UI showing detections, map, and terminal telemetry; live stream via Flask
### Highlight
- GPS --> Jetson (Processing + CV) --> UDP --> UI
- RF telemetry is maintained in parallel
- I2C/USB/UART form the internal peripheral backbone, and UDP carries time-sensitive events

## Key Specs & Design Targets
- **Flight Path**: Waypoint navigation, 9-10 ft altitude, 5 mph parameter set
- **Range (control/telemetry):** ~427 ft effective radius (validated)
- **Vision performance:** 20-25 FPS camera output; <= 800 ms end-to-end latency target
- **Detection Range:** ~50 m (person)
- **Power/Weight:** 4S 3300 mAh LiPo; total takeoff weight ~2.5 kg (5.5 lb)
- **Alert cadence:** 2 Hz SOS updates on detection

## Demos
Project clips:
- Flight & platform overview: https://youtu.be/dsUHsBSHA0g
- Ground-station & detections: https://youtu.be/vNRCC04JXCo

