import cv2

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()

        image = cv2.resize(image, (500, 500))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret1, adaptive_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image = cv2.medianBlur(image, ksize=5)
        thres = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
        edges = cv2.cornerHarris(thres, 2, 3, 0.5)
        cv2.imshow('image', image)
        cv2.imshow('Threshold', adaptive_threshold)
        cv2.imshow('adaptive', thres)
        cv2.imshow('eges', edges)
        if cv2.waitKey(1) and 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
