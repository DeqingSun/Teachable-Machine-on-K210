# teachable machine - By: sundeqing - Wed Apr 29 2020

import sensor, image, time, lcd
import uos, struct, math
import KPU as kpu
paraFileName = "tm_parameter.bin"
lableFileName = "tm_labels.txt"

filesInSd = uos.listdir("/sd/")

labels = []
sampleCount = []

def getNormalizedVec(vec):
    s = 0
    for i in vec:
        s = s + i*i
    s = 1.0/math.sqrt(s)
    return [x*s for x in vec]

task = kpu.load('/sd/mbnet75_noact.kmodel')

lcd.init(freq=15000000)

if (paraFileName in paraFileName) and (lableFileName in paraFileName):
    print('parameter and label found, skip training.')
else:
    print('parameter and label not found, start training.')
    counter = 0
    labelCounter = 0
    f=open('/sd/'+paraFileName, 'wb')
    for info in uos.ilistdir('/sd/'):   #look for tm* directories
        if (info[1]==0x4000 and info[0].startswith('tm_')):
            className = info[0][3:]
            classPath = '/sd/' + info[0]
            imgFileNames = uos.listdir(classPath)
            for imgFileName in imgFileNames:
                if not imgFileName.startswith('.'): #avoid trash on mac
                    myImage = image.Image(classPath+"/"+imgFileName)
                    lcd.display(myImage,oft=(0,0))
                    lcd.draw_string(0, 0, "Training "+imgFileName)
                    myImage.pix_to_ai()
                    fmap = kpu.forward(task, myImage)
                    del myImage
                    data=fmap[:]
                    data=getNormalizedVec(data)
                    for i in range(20):
                        f.write(struct.pack('50f', *data[i*50:(i+1)*50]))
                    counter+=1

            labels.append(className)
            sampleCount.append(counter)
            labelCounter+=1
    f.close()
    f=open('/sd/'+lableFileName, 'w')
    for i in range(labelCounter):
        f.write(labels[i]+'\n'+str(sampleCount[i])+'\n')
    f.close()

