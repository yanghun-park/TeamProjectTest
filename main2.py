import numpy as np
import cv

# 허프 원 검출을 사용하여 공을 검출하는 함수
def searchCircle(frame, img_back, hsv_l, hsv_u): # 공을 찾는 함수
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # HSV 영역으로 변환
    img_range = cv2.inRange(img_hsv, hsv_l, hsv_u) # 특정 영역의 색상 추출
    img_gb = cv2.GaussianBlur(img_range, (13, 13), 0) # 노이즈 제거 및 엣지 검출에 용이하기 위해 가우시안 필터 적용

    param2_num = 20 # param2 초기값

    while True: # param2 값을 조정하기 위한 While문
        circles = cv2.HoughCircles(img_gb, cv2.HOUGH_GRADIENT, 1, 30, param1=60, param2=param2_num, minRadius=0, maxRadius=100) # 허프 원 변환을 이용하여 원 검출
        print(param2_num)
        circleCount = 0

        if circles is not None:
            circles = np.uint16(np.around(circles)) # 원들 좌표를 반올림

            for i in circles[0, :]: # 검출된 원의 개수만큼 반복
                circleCount += 1 # 원 개수 카운팅

            if circleCount < circleCountMax: # 만약 검출된 원이 최대 원 개수 미만이면
                for i in circles[0, :]:
                    cv2.circle(img_back, (i[0], i[1]), i[2], (255, 255, 0), 2)
                print("원의 개수 : ", circleCount)
                return img_back, circleCount
        else:
            print("원을 찾을 수 없음")

        if param2_num > 50:
            print("원을 찾을 수 없어 프로그램을 종료합니다. ")
            break
        else:
            param2_num += 1

    return img_back, circleCount


# 이미지 중앙값 색 검출을 위한 함수
def colorCapture(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = int(img_hsv[360][640][0])
    s = int(img_hsv[360][640][1])
    v = int(img_hsv[360][640][2])

    return h, s, v


def main():
    camera = cv2.VideoCapture(cv2.CAP_DSHOW)
    camera.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while (camera.isOpened()):
        ret, frame = camera.read()
        cv2.circle(frame, (640, 360), 10, (255, 255, 0), 2)

        cv2.imshow('Circle', frame)  # 출력

        if cv2.waitKey(1) == ord('a'):
            img_back = np.zeros((720, 1280, 3), np.uint8)  # 원 표시용 빈 이미지 생성
            h, s, v = colorCapture(frame)
            img_back, count = searchCircle(frame, img_back, (h-10, s-50, v-50), (h+10, s+50, v+50))

            text = "Found Ball : " + str(count)
            cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
            cv2.imshow('H-Circle', cv2.add(frame, img_back)) # 출력
            print("원의 총 개수 : ", count)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

