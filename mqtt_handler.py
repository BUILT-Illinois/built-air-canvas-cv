import json
import time
from awscrt import mqtt
from awsiot import mqtt_connection_builder

class MQTTHandler:
    def __init__(self, endpoint, cert_path, key_path, ca_path, client_id, topic_prefix):
        self.endpoint = endpoint
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path
        self.client_id = client_id
        self.topic = f"{topic_prefix}/data"
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