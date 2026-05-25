# 🕺 JD-Kinect

**JD-Kinect** is a lightweight, open-source bridge that turns your **Xbox Kinect v1** (Xbox 360) sensor into a full-body tracking virtual controller for Just Dance. 

By leveraging the official Kinect for Windows SDK v1.8, JD-Kinect captures true 3D skeletal motion data and streams it directly to the game. This provides a completely hands-free, immersive dancing experience without the need to hold a smartphone or joycon!

---

## ✨ Features

- **Hands-Free Dancing:** Ditch the phone. Your body is the controller.
- **True 3D Tracking:** Utilizes the Kinect's depth sensor for real metric tracking, providing superior Z-axis accuracy compared to standard webcam solutions.
- **Visual Overlay:** Includes an OpenCV-based real-time preview of your camera feed with a skeletal tracking overlay and a dynamic distance meter to help you find the optimal tracking range.
- **Low Latency 60Hz Output:** Interpolates the Kinect's 30Hz skeleton data to stream ultra-smooth 60Hz controller telemetry to the game.
- **Advanced Forearm Tracking:** Blends pure wrist position with elbow-to-wrist rotational tracking to capture the exact flicks and angular gestures required to hit "Megastar".

## 🛠️ Prerequisites

### Hardware
* **Kinect for Xbox 360** (Model 1414) OR **Kinect for Windows** (Model 1473)
* **Kinect Power Adapter** (Required; the sensor will not work on USB power alone)
* A PC running **Windows 10 or 11 (x64)**

### Software
* Just Dance 2017 (PC) or other compatible versions that support the "Play with Smartphone" feature
* [Python 3.10+](https://www.python.org/downloads/)

---

## 🚀 Installation & Setup

### 1. Install Kinect Drivers
Before touching Python, you **must** install the official Microsoft drivers in this exact order to prevent conflicts:

1. Download and install the **Kinect for Windows SDK v1.8**.
2. Download and install the **Kinect for Windows Runtime v1.8**.
3. Plug in your Kinect (Power first, then USB) and verify it appears in your Windows Device Manager without any errors.

*(Optional but recommended: Run the "Skeletal Viewer" sample provided with the SDK to confirm your Kinect is successfully tracking your skeleton before proceeding).*

### 2. Setup the Python Environment

Clone this repository and open a terminal in the project directory. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Alternatively, you can manually run `pip install websockets opencv-python numpy future`)*

### 3. Let's Dance!

1. Open Just Dance and navigate to the **"Play with your smartphone"** screen.
2. Run the controller script:
   ```bash
   python main.py
   ```
3. A window will pop up showing your camera feed. Stand roughly **1.5 to 2.5 meters** away from the sensor. Use the built-in distance meter on the screen to find the "Optimal" spot.
4. Once you see the orange skeleton overlay on your body, the app is successfully tracking you and streaming data to the game.

---

## ⚙️ Configuration & Tuning

If you feel the game is not scoring your moves correctly, you can adjust the tracking sensitivity by modifying constants inside `main.py`:

- `ACCEL_SCALE`: (Default: `5.5`) Increase this if your big moves are scoring poorly.
- `SMOOTH_ALPHA`: (Default: `0.4`) Adjust the exponential smoothing filter. Lower is smoother but adds tracking latency.
- `ROTATION_WEIGHT`: (Default: `0.35`) The amount of emphasis given to forearm rotation (elbow-to-wrist flicks) over pure wrist translation.
- `POSITION_JUMP_THRESHOLD`: (Default: `0.5`) Guards against tracking glitches by discarding frames where the skeleton teleports implausibly far.
- `KINECT_Z_CENTER`: (Default: `1.5`) The expected depth in meters where you stand.
- `KINECT_X_RANGE` / `KINECT_Y_RANGE`: (Default: `0.6`) The bounds for your arm's lateral/vertical range.

## ❓ FAQ & Troubleshooting

- **I can't navigate the game menus / change dancers!** 
  *You will still need to connect a smartphone or use standard controls to navigate the game menus before a song.*
- **The camera window says "Skeleton undetected".** 
  *Make sure you are standing within the Kinect's sweet spot (1.5m - 2.5m) and that your full upper body is visible to the sensor.*
- **Is multiplayer supported?** 
  *Currently, the script only tracks the first skeleton it finds. Multiplayer is possible but requires modifications to the code.*

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request if you want to improve tracking accuracy, add multiplayer, or enhance the quaternion math.

*Based on original concepts from the JDWebcam project by Comera.*
