import multiprocessing
from multiprocessing import Process
import predict2
import wav_to_mel
import recording
import warnings
import sen_recognition
import delete

warnings.filterwarnings('ignore')

p1 = Process(target=recording.recording)
p2 = Process(target=sen_recognition.sen_recognition)
p3 = Process(target=wav_to_mel.wav_to_mel)
p4 = Process(target=predict2.predict)
p5 = Process(target=delete.delete)

p1.start()
print("p1 start")
p2.start()
print("p2 start")
p3.start()
print("p3 start")
p4.start()
print("p4 start")
p5.start()
print("p5 start")

p1.join()
p2.join()
p3.join()
p4.join()
p5.join()

print("complete")