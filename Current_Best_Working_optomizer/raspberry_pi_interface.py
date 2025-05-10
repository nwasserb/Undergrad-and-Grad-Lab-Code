#!/usr/bin/env python3
import RPi.GPIO as GPIO
import socket
import sys

KeyboardInterrupt



# ---------------------------
# GPIO Setup
# ---------------------------
GPIO.setmode(GPIO.BCM)          # Use Broadcom pin numbering
TTL_PIN = 18                    # Change this to the pin connected to the TTL pulse
GPIO.setup(TTL_PIN, GPIO.IN)    # Set the TTL_PIN as an input

def get_state():
    """
    Reads the TTL pulse on TTL_PIN.
    Returns:
        "H" if the GPIO input is HIGH,
        "D" if the GPIO input is LOW.
    """
    if GPIO.input(TTL_PIN):
        return "H"
    else:
        return "D"

# ---------------------------
# TCP Server Setup
# ---------------------------
HOST = ''         # Listen on all interfaces
PORT = 12345      # Port to listen on (must match the port in your client code)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)  # Allow up to 5 queued connections
print(f"Server listening on port {PORT}...")

try:
    while True:
        # Accept a new connection
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        try:
            # Receive data from the client (up to 1024 bytes)
            data = client_socket.recv(1024).decode().strip()
            print(f"Received command: '{data}'")
            if data == "GET_STATE":
                # Read the current state from the TTL input
                state = get_state()
                print(f"Sending state: {state}")
                client_socket.sendall(state.encode())
            else:
                print("Received invalid command")
                client_socket.sendall("INVALID_COMMAND".encode())
        except Exception as e:
            print("Error handling client request:", e)
        finally:
            client_socket.close()
except KeyboardInterrupt:
    print("Server shutting down...")
except Exception as e:
    print("Server error:", e)
finally:
    server_socket.close()
    GPIO.cleanup()
    sys.exit(0)
