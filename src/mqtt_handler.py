import json
import time
import threading
from typing import Callable, Optional, Dict
from awscrt import mqtt
from awsiot import mqtt_connection_builder

class MQTTHandler:
    """MQTT Publisher - publishes CV/hand tracking data to AWS IoT Core."""

    def __init__(self, endpoint, cert_path, key_path, ca_path, client_id, topic):
        self.endpoint = endpoint
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path
        self.client_id = client_id
        self.topic = topic
        self.connection = None
        self.connected = False

    def connect(self): #connect to AWS
        try:
            print(f"Connecting to IOT Core: {self.endpoint}")

            self.connection = mqtt_connection_builder.mtls_from_path(
                endpoint=self.endpoint,
                cert_filepath=self.cert_path,
                pri_key_filepath=self.key_path,
                client_id=self.client_id,
                ca_filepath=self.ca_path,
                clean_session=False,
                keep_alive_secs=30
            )

            connect_future = self.connection.connect()
            connect_future.result()  # Wait for connection to complete

            self.connected = True
            print("MQTT connected to IOT Core")

        except Exception as e:
            print(f"MQTT connection error: {e}")
            self.connected = False

    def publish(self, data):
        if not self.connected or not self.connection:
            return

        try:
            message = json.dumps({
                "timestamp": int(time.time() * 1000),
                "client_id": self.client_id,
                "data": data
            })

            self.connection.publish(
                topic=self.topic,
                payload=message,
                qos=mqtt.QoS.AT_LEAST_ONCE
            )
        except Exception as e:
            print(f"MQTT publish error: {e}")
            self.connected = False

    def disconnect(self):
        if self.connection:
            try:
                disconnect_future = self.connection.disconnect()
                disconnect_future.result()  # Wait for disconnection to complete
                print("MQTT disconnected")
            except Exception as e:
                print(f"MQTT disconnection error: {e}")

        self.connected = False


class MQTTSubscriber:
    """
    MQTT Subscriber - subscribes to multiple topics and dispatches to callbacks.
    Designed for receiving both hand tracking and IMU data.
    """

    def __init__(self, endpoint, cert_path, key_path, ca_path, client_id):
        self.endpoint = endpoint
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path
        self.client_id = client_id
        self.connection = None
        self.connected = False
        self.msg_count = 0
        self.lock = threading.Lock()
        self.subscriptions = {}  # topic -> callback mapping

    def connect(self):
        """Connect to AWS IoT Core."""
        try:
            print(f"[MQTT SUB] Connecting to {self.endpoint} ...")
            self.connection = mqtt_connection_builder.mtls_from_path(
                endpoint=self.endpoint,
                cert_filepath=self.cert_path,
                pri_key_filepath=self.key_path,
                ca_filepath=self.ca_path,
                client_id=self.client_id,
                clean_session=True,
                keep_alive_secs=30,
            )
            self.connection.connect().result(timeout=10)
            self.connected = True
            print("[MQTT SUB] Connected!")
        except Exception as e:
            print(f"[MQTT SUB] Connection failed: {e}")
            self.connected = False

    def subscribe(self, topic: str, callback: Callable):
        """
        Subscribe to a topic with a callback.
        Callback signature: callback(topic: str, payload: dict)
        """
        if not self.connected:
            print(f"[MQTT SUB] Cannot subscribe to {topic} - not connected")
            return

        def _on_message(topic, payload, **kwargs):
            """Internal callback wrapper that parses JSON and calls user callback."""
            try:
                msg = json.loads(payload)
                with self.lock:
                    self.msg_count += 1
                callback(topic, msg)
            except json.JSONDecodeError as e:
                print(f"[MQTT SUB] Bad JSON on {topic}: {e}")
            except Exception as e:
                print(f"[MQTT SUB] Callback error on {topic}: {e}")

        try:
            subscribe_future, _ = self.connection.subscribe(
                topic=topic,
                qos=mqtt.QoS.AT_LEAST_ONCE,
                callback=_on_message,
            )
            subscribe_future.result(timeout=10)
            self.subscriptions[topic] = callback
            print(f"[MQTT SUB] Subscribed to: {topic}")
        except Exception as e:
            print(f"[MQTT SUB] Subscribe failed for {topic}: {e}")

    def disconnect(self):
        """Disconnect from AWS IoT Core."""
        if self.connection:
            try:
                self.connection.disconnect().result()
                print("[MQTT SUB] Disconnected")
            except Exception:
                pass
        self.connected = False
    
def format_hand_data(hand_landmarks, frame_width, frame_height, color_index, color_names, is_drawing):
        landmarks = []
        for hand in hand_landmarks:
            for lm in hand:
                landmarks.append({
                    "x": round(lm.x * frame_width, 2),
                    "y": round(lm.y * frame_height, 2),
                    "z": round(lm.z, 4)
                })
        return {
            "landmarks": landmarks,
            "color": color_names[color_index],
            "is_drawing": is_drawing
        }   
