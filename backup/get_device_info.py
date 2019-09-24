# A simple class to know about your cuda devices
import pycuda.driver as cuda
import pycuda.autoinit  # Necessary for using its functions

cuda.init()  # Necesarry for using its functions


class aboutCudaDevices():
    def __init__(self):
        pass

    def num_devices(self):
        """Return number of devices connected."""
        return cuda.Device.count()

    def devices(self):
        """Get info on all devices connected."""
        num = cuda.Device.count()
        print("%d device(s) found:" % num)
        for i in range(num):
            print(cuda.Device(i).name(), "(Id: %d)" % i)

    def mem_info(self):
        """Get available and total memory of all devices."""
        available, total = cuda.mem_get_info()
        print("Available: %.2f GB\nTotal:     %.2f GB" % (available / 1e9, total / 1e9))

    def attributes(self, device_id=0):
        """Get attributes of device with device Id = device_id"""
        return cuda.Device(device_id).get_attributes()

    def __repr__(self):
        """Class representation as number of devices connected and about them."""
        num = cuda.Device.count()
        string = ""
        string += ("%d device(s) found:\n" % num)
        for i in range(num):
            string += ("    %d) %s (Id: %d)\n" % ((i + 1), cuda.Device(i).name(), i))
            string += ("          Memory: %.2f GB\n" % (cuda.Device(i).total_memory() / 1e9))
        return string


# You can print output just by typing its name (__repr__):
aboutCudaDevices()
# 1 device(s) found:
#    1) Tesla K80 (Id: 0)
#          Memory: 12.00 GB