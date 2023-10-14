# Image mock for scanning process. Note that if this variable is 'None', then
# the scanning will not be performed from a real device.
# SCAN_IMAGE_MOCK = None
SCAN_IMAGE_MOCK = "test/images/image-1.png"

# Scanning options, such as the scanning device. Note that specifying the
# device with this parameter initiates the scanning process earlier, but it's
# optional.
OPTIONS = ""

# Scanned image output.
SCANNED_IMAGE_PATH = "/tmp/image.png"

# Command to scan from the flatbed. Note that the 'y' parameter isn't required
# when the paper size is A4.
SCAN_COMMAND = f"scanimage --mode=Color --resolution=300 --format=png -o {SCANNED_IMAGE_PATH} {OPTIONS}"

# Command to scan from the ADF. The 'y' parameter is required because, without
# it, the scanner will add white pixels to complete the image after the end of
# an A4 paper.
SCAN_COMMAND_ADF = f"{SCAN_COMMAND} --source='ADF' -y 300"

# Time to wait (in seconds) to finish the scanning process.
SCAN_TIMEOUT = 45
