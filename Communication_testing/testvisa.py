import serial.tools.list_ports

def find_visa_name(com_port):
    visa_name = None
    for port in serial.tools.list_ports.comports():
        if com_port in port.device:
            visa_name = port.device
            break
    return visa_name

# Example usage
com_port = "COM3"
visa_name = find_visa_name(com_port)
print("VISA name for COM3:", visa_name)
