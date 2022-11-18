import cv2


def decode(filename):
	img = cv2.imread(filename)
	detector = cv2.QRCodeDetector()
	data, bbox, straight_qrcode = detector.detectAndDecode(img)
	print(data)
