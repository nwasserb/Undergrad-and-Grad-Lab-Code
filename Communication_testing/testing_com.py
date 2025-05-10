import serial

# COM port settings
port = 'COM3'    # Change to the appropriate COM port
baudrate = 9600  # Baud rate

# Data to send
data_to_send = "V1,1000"

try:
    # Open the serial port
    ser = serial.Serial(port, baudrate)
    
    # Write the data
    ser.write(data_to_send.encode('utf-8'))  # Encode the string as UTF-8 before sending
    
    print("Data sent successfully")
    
    # Close the serial port
    ser.close()
    
except Exception as e:
    print("Error:", e)
