import asyncio
import websockets
import json
import time
import threading
from typing import Optional
import cv2
import pykinect
from pykinect import nui
from pykinect.nui import JointId
import numpy as np
import math
import socket
import sys

#patch for websocket limit (v1.1)
try:
    websockets.connect = websockets.client.connect = (lambda c: (lambda *a, **k: c(*a, max_size=k.get('max_size', 10*1024*1024), **k)))(websockets.connect)
except:
    pass

WS_SUBPROTOCOL = 'v1.phonescoring.jd.ubisoft.com'
ACCEL_ACQUISITION_FREQ_HZ = 200.0
ACCEL_ACQUISITION_LATENCY = 0.0
ACCEL_MAX_RANGE = 8.0
FRAME_DURATION = 0.015

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
FPS_SMOOTH = 0.12
KINECT_X_RANGE = 0.6
KINECT_Y_RANGE = 0.6
KINECT_Z_CENTER = 1.5
KINECT_Z_RANGE = 1.0
ACCEL_SCALE = 40.0
Y_AXIS_SCALE = 1.8 
WIIMOTE_LENGTH = 40 
WIIMOTE_WIDTH = 10 

COLOR_ORANGE = (0, 165, 255)

DEBUG_MODE = '--debug' in sys.argv
if DEBUG_MODE:
    sys.argv.remove('--debug')

TEST_MODE = '--test' in sys.argv
if TEST_MODE:
    sys.argv.remove('--test')

class VirtualController:

    def __init__(self, console_ip: str):
        self.pairing_url = f"ws://{console_ip}:8080/smartphone"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.disconnected = False
        
        self.should_start_accelerometer = False
        self.number_of_accels_sent = 0
        
        self.accel_lock = threading.Lock()
        self.accel_data_buffer = []

    async def send_message(self, __class: str, data: dict = {}):
        if self.ws and not self.disconnected:
            msg = {'root': {'__class': __class}}
            if data:
                msg['root'].update(data)
            
            try:
                await self.ws.send(json.dumps(msg, separators=(',', ':')))
            except Exception:
                self.disconnected = True

    async def on_message(self, raw_message: str):
        try:
            message = json.loads(raw_message)
            if '__class' not in message:
                return

            __class = message.get('__class')

            if __class == 'JD_PhoneDataCmdHandshakeContinue':
                await self.send_message('JD_PhoneDataCmdSync', {'phoneID': message['phoneID']})
            
            elif __class == 'JD_PhoneDataCmdSyncEnd':
                await self.send_message('JD_PhoneDataCmdSyncEnd', {'phoneID': message['phoneID']})

            elif __class == 'JD_EnableAccelValuesSending_ConsoleCommandData':
                self.should_start_accelerometer = True
                self.number_of_accels_sent = 0
            
            elif __class == 'JD_DisableAccelValuesSending_ConsoleCommandData':
                self.should_start_accelerometer = False
                with self.accel_lock:
                    self.accel_data_buffer.clear()
            
        except Exception:
            pass

    async def send_hello(self):
        await self.send_message('JD_PhoneDataCmdHandshakeHello', {
            'accelAcquisitionFreqHz': ACCEL_ACQUISITION_FREQ_HZ,
            'accelAcquisitionLatency': ACCEL_ACQUISITION_LATENCY,
            'accelMaxRange': ACCEL_MAX_RANGE,
        })

        try:
            async for message in self.ws:
                await self.on_message(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.disconnected = True

    async def tick_accelerometer(self):
        while not self.disconnected:
            if self.should_start_accelerometer:
                accel_data_to_send = []
                
                with self.accel_lock:
                    if self.accel_data_buffer:
                        accel_data_to_send = self.accel_data_buffer.copy()
                        self.accel_data_buffer.clear()
                
                if accel_data_to_send:
                    await self.send_message('JD_PhoneScoringData', {
                        'accelData': accel_data_to_send,
                        'timeStamp': self.number_of_accels_sent,
                    })
                    self.number_of_accels_sent += len(accel_data_to_send)
                
                await asyncio.sleep(FRAME_DURATION * 3)
            else:
                await asyncio.sleep(0.1)
        
        if self.ws and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def connect(self):
        try:
            async with websockets.connect(
                self.pairing_url,
                subprotocols=[WS_SUBPROTOCOL],
                ping_timeout=None
            ) as websocket:
                self.ws = websocket
                
                await asyncio.gather(
                    self.send_hello(),
                    self.tick_accelerometer()
                )
        
        except (OSError, websockets.exceptions.InvalidURI, websockets.exceptions.InvalidHandshake) as e:
            raise ConnectionError(f"Falha ao conectar: {e}")
        except Exception:
            pass
        finally:
            self.disconnected = True

def skeleton_to_color(kinect, joint):
    '''Convert skeleton 3D joint to color frame pixel coords'''
    try:
        x, y = kinect.skeleton_engine.skeleton_to_color(joint)
        return int(x), int(y)
    except Exception:
        return None

def kinect_accelerometer_thread(controller: VirtualController):
    window = 'Controlador Kinect Just Dance'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, TARGET_WIDTH, TARGET_HEIGHT)
    
    shared_data = {
        'last_pos_vec': None,
        'jd_x': 0.0,
        'jd_y': 0.0,
        'jd_z': 0.0,
        'has_hand': False,
        'joints': None,
        'last_tracking_state': None
    }
    fps = 0.0
    last_time = time.time()
    color_frame = [None]
    
    try:
        kinect = nui.Runtime()
    except Exception as e:
        print(f'Kinect init failed: {e}')
        controller.disconnected = True
        return
        
    # Enable color + skeleton streams
    kinect.video_stream.open(
        nui.ImageStreamType.Video, 2,
        nui.ImageResolution.Resolution640x480,
        nui.ImageType.Color
    )
    kinect.skeleton_engine.enabled = True
    
    def video_frame_ready(frame):
        data = frame.image.bits
        arr = np.frombuffer(data, dtype=np.uint8)
        arr = arr.reshape((TARGET_HEIGHT, TARGET_WIDTH, 4))
        color_frame[0] = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            
    kinect.video_frame_ready += video_frame_ready
    
    def skeleton_frame_ready(frame):
        has_hand = False
        
        for skeleton in frame.SkeletonData:
            if skeleton.eTrackingState == nui.SkeletonTrackingState.NOT_TRACKED:
                continue
                
            if DEBUG_MODE and skeleton.eTrackingState != shared_data['last_tracking_state']:
                state_names = {
                    nui.SkeletonTrackingState.POSITION_ONLY: "POSITION_ONLY",
                    nui.SkeletonTrackingState.TRACKED: "TRACKED"
                }
                print(f"[DEBUG] Skeleton detected! State: {state_names.get(skeleton.eTrackingState, skeleton.eTrackingState)}")
                shared_data['last_tracking_state'] = skeleton.eTrackingState

            if skeleton.eTrackingState != nui.SkeletonTrackingState.TRACKED:
                continue
                
            joints = skeleton.SkeletonPositions
            wrist = joints[JointId.WristRight]
            
            # Normalize to approx -1..+1 range
            dx = wrist.x / KINECT_X_RANGE
            dy = wrist.y / KINECT_Y_RANGE
            dz = (wrist.z - KINECT_Z_CENTER) / KINECT_Z_RANGE
            
            pos_vec = np.array([dx, dy, dz], dtype=np.float32)
            
            if shared_data['last_pos_vec'] is not None:
                raw_accel = (pos_vec - shared_data['last_pos_vec']) * ACCEL_SCALE
                accel = np.clip(raw_accel, -4.0, 4.0)
                
                base_x = pos_vec[0] * 1.5
                base_y = pos_vec[1] * 1.5
                xy_mag_sq = base_x**2 + base_y**2
                base_z = -math.sqrt(max(0.0, 1.0 - xy_mag_sq))
                
                jd_x = float(np.clip(base_z + accel[2], -8.0, 8.0))
                jd_y = float(np.clip(base_x + accel[0], -8.0, 8.0))
                jd_z = float(np.clip(base_y + accel[1], -8.0, 8.0))
                
                shared_data['jd_x'] = jd_x
                shared_data['jd_y'] = jd_y
                shared_data['jd_z'] = jd_z
                
                if controller.should_start_accelerometer:
                    with controller.accel_lock:
                        t = (jd_x, jd_y, jd_z)
                        controller.accel_data_buffer.extend([t, t, t])
                        
            shared_data['last_pos_vec'] = pos_vec.copy()
            shared_data['has_hand'] = True
            shared_data['joints'] = joints
            has_hand = True
            break
            
        if not has_hand:
            if shared_data['last_tracking_state'] is not None and shared_data['last_tracking_state'] != nui.SkeletonTrackingState.NOT_TRACKED:
                if DEBUG_MODE:
                    print("[DEBUG] Skeleton lost.")
                shared_data['last_tracking_state'] = nui.SkeletonTrackingState.NOT_TRACKED
            
            shared_data['last_pos_vec'] = None
            shared_data['jd_x'] = 0.0
            shared_data['jd_y'] = 0.0
            shared_data['jd_z'] = 0.0
            shared_data['has_hand'] = False
            shared_data['joints'] = None

    kinect.skeleton_frame_ready += skeleton_frame_ready
    
    while not controller.disconnected:
        now = time.time()
        
        has_hand = shared_data['has_hand']
        joints = shared_data['joints']
        jd_x = shared_data['jd_x']
        jd_y = shared_data['jd_y']
        jd_z = shared_data['jd_z']
        
        # Draw overlay on color frame
        frame = color_frame[0]
        if frame is None:
            cv2.waitKey(1)
            time.sleep(0.005)
            continue
            
        frame = frame.copy()
        
        if has_hand and frame is not None:
            wrist_px = skeleton_to_color(kinect, joints[JointId.WristRight])
            shoulder_px = skeleton_to_color(kinect, joints[JointId.ShoulderRight])
            hand_px = skeleton_to_color(kinect, joints[JointId.HandRight])
            
            if wrist_px and shoulder_px:
                cv2.line(frame, shoulder_px, wrist_px, COLOR_ORANGE, 2, cv2.LINE_AA)
            if wrist_px:
                cv2.circle(frame, wrist_px, 6, COLOR_ORANGE, -1, cv2.LINE_AA)
            if hand_px:
                cv2.circle(frame, hand_px, 4, (0,255,255), -1, cv2.LINE_AA)
        
        delta_time = now - last_time
        if delta_time > 0:
            inst_fps = 1.0 / delta_time
            fps = (1-FPS_SMOOTH)*fps + FPS_SMOOTH*inst_fps if fps > 0 else inst_fps
        last_time = now
        
        y_off = 30
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1, cv2.LINE_AA)
        y_off += 25
        
        if not has_hand:
            cv2.putText(frame, 'Skeleton undetected', (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1, cv2.LINE_AA)
            y_off += 25
            
        cv2.putText(frame, f'JD-> X:{jd_x:+.1f} Y:{jd_y:+.1f} Z:{jd_z:+.1f}',
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1, cv2.LINE_AA)
        y_off += 25
        cv2.putText(frame, 'By Comera', (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 1, cv2.LINE_AA)
                    
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        closed = False
        try:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                closed = True
        except cv2.error:
            closed = True
            
        if key == ord('q') or closed:
            controller.disconnected = True
            break
            
    kinect.close()
    cv2.destroyAllWindows()
    controller.disconnected = True

def get_local_ip_prefix():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1.0)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        prefix = ".".join(ip.split('.')[:-1]) + "."
        return prefix
    except Exception:
        return "192.168.1."
    finally:
        s.close()

async def find_console_ip(prefix: str):
    tasks = []
    for i in range(100, 151):
         ip = f"{prefix}{i}"
         tasks.append(check_ip(ip))
    for i in list(range(1, 100)) + list(range(151, 255)):
        ip = f"{prefix}{i}"
        tasks.append(check_ip(ip))
    
    results = await asyncio.gather(*tasks)
    found_ips = [ip for ip in results if ip]
    
    if found_ips:
        return found_ips[0]
    else:
        common_prefixes = ["192.168.0.", "192.168.15.", "10.0.0."]
        if prefix in common_prefixes: common_prefixes.remove(prefix)
        
        for next_prefix in common_prefixes:
            tasks = []
            for i in range(1, 255):
                ip = f"{next_prefix}{i}"
                tasks.append(check_ip(ip))
            results = await asyncio.gather(*tasks)
            found_ips = [ip for ip in results if ip]
            if found_ips:
                return found_ips[0]

    return None

async def check_ip(ip: str):
    try:
        url = f"ws://{ip}:8080/smartphone"
        await asyncio.wait_for(
            websockets.connect(url, subprotocols=[WS_SUBPROTOCOL], open_timeout=0.4, close_timeout=0.1),
            timeout=0.5
        )
        return ip
    except Exception:
        return None

async def async_main():
    if TEST_MODE:
        print("Running in TEST MODE. No console connection will be made.")
        class MockController:
            def __init__(self):
                self.disconnected = False
                self.should_start_accelerometer = False
                self.accel_lock = threading.Lock()
                self.accel_data_buffer = []
        controller = MockController()
        t_webcam = threading.Thread(target=kinect_accelerometer_thread, args=(controller,), daemon=True)
        t_webcam.start()
        try:
            while t_webcam.is_alive():
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            controller.disconnected = True
        return

    console_ip = None
    
    if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
        console_ip = sys.argv[1][2:]
    
    if not console_ip:
        print("Searching for Just Dance console on the network...")
        ip_prefix = get_local_ip_prefix()
        console_ip = await find_console_ip(ip_prefix)
    
    if not console_ip:
        print("Error: Could not find the game console.")
        print("Make sure Just Dance is running on the 'Play with your smartphone' screen and connected to the same network.")
        return

    print(f"Found console at {console_ip}! Connecting...")
    controller = VirtualController(console_ip)

    t_webcam = threading.Thread(target=kinect_accelerometer_thread, args=(controller,), daemon=True)
    t_webcam.start()

    try:
        await controller.connect()
    except ConnectionError:
        controller.disconnected = True
    
    if t_webcam.is_alive():
        t_webcam.join(timeout=2.0)

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
