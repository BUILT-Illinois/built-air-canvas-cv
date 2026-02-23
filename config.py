#configs
#Websocket configuration

DEVICE_ID = "laptop_01"  # Change for each device, to know which device is drawing
SEND_INTERVAL_MS = 100  # Send data every 100ms (10 times/second)


WEBSOCKET_ENABLED = False #websocket toggle
WEBSOCKET_URL = "wss://gs84tnv26l.execute-api.us-east-2.amazonaws.com/production/" 

MQTT_ENABLED = True #mqtt toggle
MQTT_ENDPOINT = "aevqdnds5bghe-ats.iot.us-east-2.amazonaws.com"
MQTT_CERT_PATH = "certs/device-certificate.pem.crt" #replace device with certificate
MQTT_KEY_PATH = "certs/device-private.pem.key" #replace device with key
MQTT_CA_PATH = "certs/AmazonRootCA1.pem"

MQTT_TOPIC_PREFIX = f"handtracking/{DEVICE_ID}"
