import socket
import serial

def setup_ethernet_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Listening for commands on {host}:{port}...")

    return server_socket


def send_command_to_device(command):
    # Example: open USB serial port (replace 'COMPORT' with your USB COM port)
    ser = serial.Serial('COM3', 9600)
    ser.write(command.encode('ascii'))
    ser.close()
    print("Command sent to device")

def receive_command_from_client(server_socket):
    client_socket, addr = server_socket.accept()
    command = client_socket.recv(1024).decode("utf-8")
    client_socket.close()  # Close the client socket after receiving the command
    print(f"Received command from Computer 1: {command}")
    print(command.encode("utf-8"))
    send_command_to_device(command)
    return command

if __name__ == "__main__":
    host = "10.0.3.12"  # Listen on all available interfaces
    port = 12345  # Choose a port for Ethernet communication

    server_socket = setup_ethernet_server(host, port)

    while True:
        command = receive_command_from_client(server_socket)
        # Forward the command to the USB device (not implemented here)

    server_socket.close()
