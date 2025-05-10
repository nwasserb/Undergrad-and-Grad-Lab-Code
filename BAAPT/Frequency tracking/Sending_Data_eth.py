import socket
import sys

def send_command_to_client(host, port, command):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    client_socket.sendall(command.encode("utf-8"))
    client_socket.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <command>")
        sys.exit(1)

    computer_2_host = "10.0.3.11"  # Replace with the IP address of Computer 2
    computer_2_port = 12345  # Choose a port for communication with Computer 2

    command = sys.argv[1]

    send_command_to_client(computer_2_host, computer_2_port, command)



