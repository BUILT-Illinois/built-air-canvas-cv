import json
import threading
import time
import websocket
from queue import Queue, Empty

class WebSocketHandler: 
    def __init__(self, url, device_id):
        self.url = url
        self.device_id = device_id
        self.ws = None
        self.connected = False
        self.data_queue = Queue(maxsize=100)
        self.send_thread = None
        self.running = False

    def connect(self):
        """Connect to WebSocket server"""
        try:
            print(f"Connecting to WebSocket: {self.url}")
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )

            #Runs WebSocket in thread
            self.running = True
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()

            #Start Sender Thread
            self.send_thread = threading.Thread(target=self._send_worker, daemon=True)
            self.send_thread.start()

            time.sleep(1)
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            self.connected = False

    def on_open(self, ws): #when connected
        print("WebSocket connected!")
        self.connected = True
    
    def on_message(self, ws, message): #handles incoming messages from server
        print(f"Received: {message}")

    def on_error(self, ws, error): #handles errors
        print(f"WebSocket error: {error}")
        self.connected = False

    def on_close(self, ws, close_status_code, close_msg): #when  disconnected
        print("WebSocket disconnected")
        self.connected = False
    
    def send_data(self, hand_data): #
        if not self.connected:
            return
        try:
            #Add to queue
            self.data_queue.put_nowait({
                "action": "sendData",
                "device_id": self.device_id,
                "timestamp": int(time.time() * 1000),
                "data": hand_data
            })
        except:
            #queue full, skips frame
            pass
    
    def _send_worker(self): #sends queued data
        while self.running:
            try:
                #get data from queue
                data = self.data_queue.get(timeout=0.1)

                if self.connected and self.ws:
                    try:
                        self.ws.send(json.dumps(data))
                    except Exception as e:
                        print(f"Error sending data: {e}")
                        self.connected = False
            except Empty:
                continue
            except Exception as e:
                print(f"Send worker error: {e}")
    
    def disconnect(self):
        self.running = False
        if self.ws:
            self.ws.close()
        print("WebSocket handler stopped")


def format_hand_data(hand_landmarks_list, frame_width, frame_height, color_index, colors, is_drawing):
    """Formats hand landmarks into a serializable format"""
    hands_data = []

    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        #convert landmarks to simple list
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.append({
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z)
            })
        #get index finger tip position in pixel coordinates
        index_tip = {
            "x": int(hand_landmarks[8].x * frame_width),
            "y": int(hand_landmarks[8].y * frame_height)
        }

        hands_data.append({
            "hand_id": hand_idx,
            "landmarks": landmarks,
            "index_finger_tip": index_tip,
            "is_drawing": is_drawing,
            "color": colors[color_index] if color_index < len(colors) else colors[0]
        })
    return {
        "hands": hands_data,
        "frame_dimensions": {
            "width": frame_width,
            "height": frame_height
        }
    }