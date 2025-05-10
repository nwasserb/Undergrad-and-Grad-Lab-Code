import numpy as np

def wrap_voltage(value, low=-5000, high=5000):
    if(value >= low and value <= high):
        return value
    else:
        if (value // (2*high)) == 1: 
            print("this is neg")
            return ((-np.abs(value) % 10000) * np.sign(value)) - high 
        else:
            print("this is wrong")
            return (-np.abs(value) % 5000) * np.sign(value)
        # return -5000


print(wrap_voltage(10000))