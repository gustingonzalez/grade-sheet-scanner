# Image mock for scanning process. Note that if this variable is 'None', then
# the scanning will not be performed from a real device.
# SCAN_IMAGE_MOCK = None
SCAN_IMAGE_MOCK = "test/images/image-1.png"

# Parameter to specify a device from which the image will be scanned (this can
# be obtained with 'scanimage -L'). Note that scanning works without specifying
# the device, but using this parameter initiates the process earlier.
# SCAN_DEVICE_PARAMETER = "-d '...'"
SCAN_DEVICE_PARAMETER = ""

# Scanned image output.
SCANNED_IMAGE_PATH = "/tmp/image.png"

# Command to scan from the flatbed. Note that the 'y' parameter isn't required
# when the paper size is A4.
SCAN_COMMAND = f"scanimage --mode=Color --resolution=300 --format=png -o {SCANNED_IMAGE_PATH} {SCAN_DEVICE_PARAMETER}"

# Command to scan from the ADF. The 'y' parameter is required because, without
# it, the scanner will add white pixels to complete the image after the end of
# an A4 paper.
SCAN_COMMAND_ADF = f"{SCAN_COMMAND} --source='ADF' -y 300"

# Time to wait (in seconds) to finish the scanning process.
SCAN_TIMEOUT = 45
