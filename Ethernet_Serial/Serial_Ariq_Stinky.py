import socket
TARGET_IP = '10.0.3.33'
PORT = 1236

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((TARGET_IP, PORT))

        print(f"connected to server at {TARGET_IP}:{PORT}")

        while True:
            data = input("enter data to send")

            client_socket.sendall(data.encode())

if __name__ == "__main__":
    main()