import cv2


def get_contours(canny, img):
    img_contour = img.copy()
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_cor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if obj_cor == 4:
                object_type = "Rectangle"

                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_contour, object_type,
                            (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 0, 0), 2)
    return img_contour


if __name__ == "__main__":

    cap = cv2.VideoCapture("doc_2022-12-13_16-29-58.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1)
        canny = cv2.Canny(blur, 50, 50)
        res = get_contours(canny, frame)

        cv2.imshow("Stack", res)

        key = cv2.waitKey(1)
        if key == "q":
            continue
