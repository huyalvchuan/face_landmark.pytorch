import cv2
import torch
from PIL import Image
import numpy as np
from mtcnn_pytorch.src.detector import detect_faces
from network import ONet
from dataloader import test_transform, scale
from headpose import get_head_pose


net = ONet().cuda()
net.load_state_dict(torch.load('./checkpoints/new_data_20.pkl'))


def get_landmark(img):
    h, w, _ = img.shape
    data = test_transform({'image': Image.fromarray(img[:, :, [2, 1, 0]])})['image'].unsqueeze(0).cuda()
    output = net(data)
    landmark = output[0].view(-1, 2).cpu().detach().numpy()
    p1, p2 = get_head_pose(landmark, img)
    p1 = [p1[0] * w / scale, p1[1] * h / scale]
    p2 = [p2[0] * w / scale, p2[1] * h / scale]
    return landmark * [w / scale, h / scale], p1, p2


def get_biggest_face(boxs):
    if boxs.shape[0] == 1:
        return boxs[0]
    areas = (boxs[:, 2] - boxs[:, 0]) * (boxs[:, 3] - boxs[:, 1])
    return boxs[np.argmax(areas)]


def get_faces(img):
    from PIL import Image

    image = Image.fromarray(img[:, :, [2, 1, 0]])
    bounding_boxes, landmarks = detect_faces(image)
    if bounding_boxes.shape[0] == 0:
        return "no face"
    areas = get_biggest_face(bounding_boxes)
    return areas


def cv2_cam():
    cap = cv2.VideoCapture(0)
    nums = 0
    while True:
        ret,frame = cap.read()
        areas = get_faces(frame)
        if not areas == "no face":
            face_frame = frame[int(areas[1]) + 25: int(areas[3]), int(areas[0]): int(areas[2]), :]
            landmark, p1, p2 = get_landmark(face_frame)
            landmark = landmark + [int(areas[0]), int(areas[1])+25]
            p1 = (int(p1[0]) + int(areas[0]), int(p1[1]) + int(areas[1])+25)
            p2 = (int(p1[1]) + int(areas[0]), int(p2[1]) + int(areas[1])+25)

            cv2.line(frame, p1, p2, (255,0,0), 2)
            cv2.rectangle(frame, (int(areas[0]), int(areas[1])), (int(areas[2]), int(areas[3])), (0, 255, 0), 1)
            landmark = landmark.tolist()
            for i in range(len(landmark)):
                l = landmark[i]
                landmark[i] = (int(l[0]), int(l[1]))
                cv2.circle(frame, landmark[i], 1, (0, 0, 255), 1)
        # cv2.imwrite(r"E:\papers\face\test1220\test_lv/{}.jpg".format(nums), frame)
        cv2.imshow('frame',frame)
        nums += 1
        if cv2.waitKey(1) &0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

cv2_cam()