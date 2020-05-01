# Teachable Machine on K210

This is a project porting [Teachable Machine](https://teachablemachine.withgoogle.com/) project to K210 platform. Instead of running in a browser, this project running on a piece of low-cost hardware.

Just like the original Teachable Machine project. No code is needed to play with this project. If you need to change sample and classifying classes, all you need to do is to put the micro-sd card into a computer, drag, and delete a few files. All training will happen the next time you power on the board, and the project will recognize new objects. 

All code of this project is open-sourced and written in Python. It would be easy for you to modify the project, change output, or connect lights, motors, etc. The hardware can be easily embedded into your project.

![Script running gif](https://raw.githubusercontent.com/DeqingSun/Teachable-Machine-on-K210/master/images/sample_loop.gif)


## What is K210?

Kendryte K210 is an edge computing system-on-chip with dual-core 64bit RISC-V CPU and neural network processor. 

Generally speaking, it is way easier to use a development board to do experiments or small projects. This project is developed and tested on a [Sipeed M1 Dock Kit](https://www.seeedstudio.com/Sipeed-M1-dock-suit-M1-dock-2-4-inch-LCD-OV2640-K210-Dev-Board-1st-RV64-AI-board-for-Edge-Computing.html) in a [3D printed case](https://www.thingiverse.com/thing:3377443).  

Although not tested yet, Other K210 boards such as [Sipeed MAIX Bit](https://www.seeedstudio.com/Sipeed-MAix-BiT-for-RISC-V-AI-IoT-1-p-2873.html), [Sipeed Maixduino](https://www.seeedstudio.com/Sipeed-Maixduino-for-RISC-V-AI-IoT-p-4046.html), or [M5StickV](https://m5stack.com/products/stickv) may also work.


## How to play with the project? 

## Link to the technical detail of Teachable Machine

<https://observablehq.com/@nsthorat/how-to-build-a-teachable-machine-with-tensorflow-js>

## What does the python code do?

Here I'll briefly summarize how Teachable Machine (V1) works. Rather than training a new artificial neural network for your samples, Teachable Machine uses a pre-trained MobileNet model that can predict a 224*224 color image, generating a probability for each of the 1000 classes.

In the training stage, the Teachable Machine runs MobileNet model on all sample images you provide and get a probability list of each image. In the predicting stage, the Teachable Machine runs MobileNet model on the input image, gets the probability list. Then the machine will compare the probability list of the unknown image to the lists of all sample images, and see which class has more similar lists. 

Back to the Python script in this project, the code will first check if ```tm_parameter.bin``` and ```tm_labels.txt``` exist on the micro SD card. If the 2 files both exist, the training process will be skipped. 

Otherwise, the script will check all folders with a name that starts with "tm_" in the card's root folder. Each folder contains samples of a certain image. The script will use the folder name as the class name and get a normalized prediction vector of each image. The result will be written to ```tm_parameter.bin``` and ```tm_labels.txt```.

Then the script will load data from ```tm_parameter.bin``` and ```tm_labels.txt```, then start camera capturing. For each image from the camera, the script will get the normalized prediction vector of the camera image. Then the script uses a dot product to calculate the similarity between the camera image and each sample image. At last, the largest 5 results are pickup, and the predicted class will be determined by the highest representation.  

## Photo of K210 detecting objects

<img src="https://raw.githubusercontent.com/DeqingSun/Teachable-Machine-on-K210/master/images/sample_empty.jpg"  width="300">
<img src="https://raw.githubusercontent.com/DeqingSun/Teachable-Machine-on-K210/master/images/sample_apple.jpg"  width="300">
<img src="https://raw.githubusercontent.com/DeqingSun/Teachable-Machine-on-K210/master/images/sample_banana.jpg"  width="300">
