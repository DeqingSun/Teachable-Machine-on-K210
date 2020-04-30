# camera for SD card - By: sundeqing - Mon Apr 20 2020

import sensor, image, time, lcd
from Maix import GPIO
from fpioa_manager import fm
from board import board_info
import uos

fm.register(board_info.BOOT_KEY, fm.fpioa.GPIO1, force=True)
input = GPIO(GPIO.GPIO1, GPIO.IN)


photoIndex = 0
prevInputValue = 1

try:
    filesInSd = uos.listdir("/sd")
    print(filesInSd)
    while(("photo%03d.jpg"%photoIndex) in filesInSd):
        photoIndex = photoIndex+1
except:
    photoIndex = "no card"

lcd.init(freq=15000000)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
sensor.set_vflip(0)
sensor.set_hmirror(0)
sensor.skip_frames(time = 1000)

clock = time.clock()

while(True):
    clock.tick()
    img = sensor.snapshot()
    inputValue = input.value()
    if (inputValue==1 and prevInputValue==0):
        img.save("/sd/photo%03d.jpg"%photoIndex, quality=95)
        photoIndex=photoIndex+1;

    prevInputValue = inputValue
    img.draw_string(2,2, str(photoIndex), color=(255,0,0), scale=2)
    lcd.display(img)

fm.unregister(board_info.BOOT_KEY, fm.fpioa.GPIO1)
