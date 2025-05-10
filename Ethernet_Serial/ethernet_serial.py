import socket
import serial

# Network settings
HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 1234  # Port number to listen on

# Serial port settings
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust this to match Raspberry Pi's serial port
BAUD_RATE = 9600

def main():
    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind the socket to the address and port
        server_socket.bind((HOST, PORT))
        
        # Listen for incoming connections
        server_socket.listen()

        print(f"TCP server is listening on {HOST}:{PORT}")

        # Accept incoming connections
        client_socket, client_address = server_socket.accept()

        # Open serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

        with client_socket, ser:
            print(f"Connected to client at {client_address}")

            while True:
                # Receive data from the client (Ethernet)
                data = client_socket.recv(1024)
                if not data:
                    break

                # Write data to the serial port
                ser.write(data)

                # Read data from the serial port
                serial_data = ser.readline()

                # Send data back to the client (Ethernet)
                client_socket.sendall(serial_data)

if __name__ == "__main__":
    main()
