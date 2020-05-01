# teachable machine - By: sundeqing - Wed Apr 29 2020

#make sure firmware has a new ulab lib that fixed dot product issue
import sensor, image, time, lcd
import uos, struct, math, array
import KPU as kpu
import ulab as np

paraFileName = "tm_parameter.bin"
lableFileName = "tm_labels.txt"

filesInSd = uos.listdir("/sd/")

labels = []
sampleCount = []

def getNormalizedVec(vec):
    s = sum([x*x for x in vec])
    s = 1.0/math.sqrt(s)
    return [x*s for x in vec]

task = kpu.load('/sd/mbnet75_noact.kmodel')

lcd.init(freq=15000000)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
sensor.set_vflip(0)
sensor.set_hmirror(0)
sensor.run(1)

if (paraFileName in filesInSd) and (lableFileName in filesInSd):
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

del filesInSd
labels = []
sampleCount = []

lcd.draw_string(236, 10, "Teachable",lcd.WHITE)
lcd.draw_string(236, 25, "Machine",lcd.WHITE)
lcd.draw_string(236, 40, "on K210",lcd.WHITE)

lcd.draw_string(236, 195, "Ported by",lcd.WHITE)
lcd.draw_string(236, 210, "Deqing",lcd.WHITE)

#load labels from file
f=open('/sd/'+lableFileName, 'r')
while True:
    nameLine = f.readline()
    if not nameLine:
        break
    countLine = f.readline()
    if not countLine:
        break
    labels.append(nameLine.strip())
    sampleCount.append(int(countLine.strip()))
f.close()

totalSample = sampleCount[-1]

#load parameter to memory for faster access
parameterList = np.zeros((totalSample,1000), dtype=np.float)
f=open('/sd/'+paraFileName, 'rb')
for j in range(totalSample):
    c = 0
    for i in range(20):
        readBuf = struct.unpack('50f',f.read(50*4))
        for k in range(50):
            parameterList[j,c]=readBuf[k]
            c+=1
f.close()

clock = time.clock()
while(True):
    img = sensor.snapshot()
    clock.tick()

    fmap = kpu.forward(task, img)
    data=np.array(getNormalizedVec(fmap[:]),dtype=np.float).transpose()

    result = np.linalg.dot(parameterList,data).flatten()

    kParameter = min(5,len(result))# 5 nearest neighbors, or total samples if less
    knnResult = sorted(range(len(result)), key=lambda x: result[x])[-kParameter:]
    fCount = [0]*len(labels)
    for n in knnResult:
        for j in range(len(fCount)):
            if (n<sampleCount[j]):
                fCount[j] += 1
                break;
    objectId = fCount.index(max(fCount))
    lcd.display(img,oft=(0,0))
    lcd.draw_string(0, 224, labels[objectId]+" "+str(int(fCount[objectId]*100/kParameter))+"%          ",lcd.WHITE)
    print("fps",clock.fps())

