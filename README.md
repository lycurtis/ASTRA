# ASTRA - Autonomous System for Tracking, Reconnaissance, and Assistance
An open-source UAV platform for search-and-rescue (SAR) that combines real-time human detection, waypoint navigation, and ground-station visualization

![1744755671148](https://github.com/user-attachments/assets/ddf777fc-4f2d-4563-b751-98230e194a4a)
![ASTRA](ASTRA.PNG)

## Table of contents
- [What is ASTRA](#what-is-astra)
- [Core Capabilities](#core-capabilities) 
- [System Architecture (High Level)](#system-architecture-high-level)
- [Key Specs & Design Targets](#key-specs--design-targets)
- [Demos](#demos)
- [How It Works (Data Flow)](#how-it-works-data-flow)
- [Testing Highlights](#testing-highlights)
- [Team](#team)

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

## How It Works (Data Flow)
1. `astra.py` boots the pipeline on the Jetson
2. `human_detect.py` processes OAK-D-Lite frames w/ YOLOv8 and logs `DETECTED`/`NOT_DETECTED`
3. `sos_transmitter.py` tails the log and transmits detection state via UDP to the ground station at the configured IP:5005 (de-duped on state change)
4. `astraui.py` from the ground station receives UDP, renders camera, map, and terminal; includes an **Offline Mode** map when the internet is unavailable

## Testing Highlights
- **RF control range:** Stable up to ~427 ft (two environments)
- **UDP comms:** Transmitter and receiver validated; terminal shows “PERSON DETECTED” / “Searching for persons…” on packets
- **Latency & video:** Bounded queue eliminated stale frames and reduced perceived lag in the UI
- **EMI mitigation:** Camera + USB 3.0 near GPS caused lock loss; shielding, grounding, cable reroute, and ferrites restored stable lock
- **Prop selection:** 1147 props chosen for best balance of lift/stability vs. 1045/1245

## Team
### Benjamin Kim
Main: Computer Vision (YOLOv8), PersonID

### Curtis Ly
Main: Hardware, System Architect 

### Pryce Matsudaira
Main: User Interface, PersonID  