import mss
import numpy


class Grabber:
    type = "mss"
    sct = mss.mss()

    def get_image(self, grab_area):
        img = numpy.array(self.sct.grab(grab_area))
        img = numpy.delete(img, 3, axis=2)
        return img

if __name__ == "__main__":
    Grabber()
