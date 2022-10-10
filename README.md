

# Automatic License Plate Detection

## Description
The Automatic License Plate Detection Application (ALPD) is a system that automatically reads and records the number plates of vehicles that enter and exit a car park. The system is designed to make the management of car parks more efficient and to reduce the chances of vehicles being stolen or illegally parked. 

Using a number of video cameras, the ALPD reads the number plates of vehicles as they enter and exit the car park. This information can be used to track the movements of vehicles in and out of the car park. The Car Park Application can be used to monitor the occupancy of a car park and to provide data on the usage of the car park. 

The Car Park Application is a valuable tool for the management of car parks. It is simple and user-friendly and can be used to improve the efficiency and reduce costs of managing car parks, allowing both a more seamless experience for customers and operators alike.


## Installation
The following instructions are for Microsoft Windows operating system: 

1. First, create a folder, tentatively titled “Tensorflow”. Feel free to go wild with the name choice though, we don’t mind.

2. Using the command prompt, cd into Tensorflow, then git clone https://github.com/tensorflow/models 

3. Create a python virtual environment. This can be done by the command “Python -m venv nameOfVirtualEnvironment”

4. Initiate the virtual environment by typing .\nameOfVirtualEnvironment\Scripts\activate into the command prompt.

5. Download Protoc from the following link: https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip 

6. Extract and copy Protoc into your created Tensorflow folder.

(If needed)  You might need the libraries numpy and wheel to run the application. If this is the case, these can be installed by typing “pip install numpy” and “pip install wheel” into the command line respectively.

7. Navigate into the research folder with the command “cd Tensorflow\Model\Research”.

8. Now the tricky bit:
Enter the following command 
“C:\your\path\to\bin\protoc.exe object_detection\protos\*.proto --python_out=. && copy object_detection\packages\tf2\setup.py setup.py && python setup.py build && python setup.py install”
“Your\path\to\” should be replaced with your path to the protoc file, for example: \Desktop\TensorFlow\bin\protoc.exe

9. This might take a while to install. Errors might occur if your machine does not have relevant libraries, simply follow the prompted guides to install them.

10. Once everything has installed successfully, enter the following command, “cd Tensorflow/models/research/slim && pip install -e .” 

11. Install our project found at https://github.com/blythjam/ITC303_Flask_APP, this can be done by copying the address found on the link (see pictured) and typing “git clone link” into the command prompt.

12. cd into ITC303_Flask_APP folder

13. Git checkout refixedUI

14. Run the application by typing “python LicensePlateDetection.py”

15. Voila! The beautiful application is running and can be found on port 5000. Refer to the following pages on how to work the app.
