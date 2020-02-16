import cv2
import torch
from PIL import Image
import numpy as np
from mtcnn_pytorch.src.detector import detect_faces
from network import ONet
from dataloader import test_transform, scale
from headpose import get_head_pose
from headstate import mouth_judge


net = ONet().cuda()
net.load_state_dict(torch.load('./checkpoints/new_data_20.pkl'))
net.eval()

def get_landmark(img):
    h, w, _ = img.shape
    data = test_transform({'image': Image.fromarray(img[:, :, [2, 1, 0]])})['image'].unsqueeze(0).cuda()
    output = net(data)
    landmark = output[0].view(-1, 2).cpu().detach().numpy()

    p1, p2, angle = get_head_pose(landmark, img)
    p1 = [p1[0] * w / scale, p1[1] * h / scale]
    p2 = [p2[0] * w / scale, p2[1] * h / scale]
    return landmark * [w / scale, h / scale], p1, p2, angle


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
    return areas, landmarks[0]


def cv2_cam():

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
 
    # out = cv2.VideoWriter('out.avi', fourcc, 20.0, (640, 480))

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    out = cv2.VideoWriter()
    out.open('out.mp4',fourcc, 10.0 ,(640, 480),True)
    nums = 0
    while True:
        ret,frame = cap.read()
        areas, five = get_faces(frame)
        if not areas == "no face":
            face_frame = frame[int(areas[1]) + 30: int(areas[3]), int(areas[0]): int(areas[2]), :]
            landmark, p1, p2, angle = get_landmark(face_frame)
            landmark = landmark + [int(areas[0]), int(areas[1])+30]
            p1 = (int(p1[0]) + int(areas[0]), int(p1[1]) + int(areas[1])+30)
            p2 = (int(p2[0]) + int(areas[0]), int(p2[1]) + int(areas[1])+30)

            cv2.line(frame, p1, p2, (255,0,0), 2)
            cv2.rectangle(frame, (int(areas[0]), int(areas[1])), (int(areas[2]), int(areas[3])), (0, 255, 0), 1)

            mouth_state = "close" if mouth_judge(landmark[51][1], landmark[62][1], landmark[66][1], landmark[57][1]) else "open"

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'mouth: {}'.format(mouth_state), (5, 100), font, 1, (255, 255, 255), 1)
            cv2.putText(frame, 'eye: {}'.format("open"), (5, 150), font, 1, (255, 255, 255), 1)
            cv2.putText(frame, 'yaw: {}'.format('%.2f' % angle[0]), (5, 200), font, 1, (255, 255, 255), 1)
            cv2.putText(frame, 'roll: {}'.format('%.2f' % angle[1]), (5, 250), font, 1, (255, 255, 255), 1)
            cv2.putText(frame, 'pitch: {}'.format('%.2f' % angle[2]), (5, 300), font, 1, (255, 255, 255), 1)

            landmark = landmark.tolist()
            for i in range(len(landmark)):
                l = landmark[i]
                landmark[i] = (int(l[0]), int(l[1]))
                cv2.circle(frame, landmark[i], 1, (0, 0, 255), 1)
            cv2.resize(frame, (640, 480))
            out.write(frame)

        # cv2.imwrite(r"E:\papers\face\test1220\test_lv/{}.jpg".format(nums), frame)
        cv2.imshow('frame',frame)
        nums += 1
        if cv2.waitKey(1) &0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def cv2_imgs():
    nums = 0
    import glob
    imgs = glob.glob(r"E:\papers\face\landmark\test_imgs\*")
    while True:
        frame = cv2.imread(imgs[nums])
        areas, five_landmark = get_faces(frame)
        if not areas == "no face":
            # face_frame = frame[int(areas[1]) + 30: int(areas[3]), int(areas[0]): int(areas[2]), :]
            # landmark, p1, p2 = get_landmark(face_frame)
            # landmark = landmark + [int(areas[0]), int(areas[1])+30]
            # p1 = (int(p1[0]) + int(areas[0]), int(p1[1]) + int(areas[1])+30)
            # p2 = (int(p2[0]) + int(areas[0]), int(p2[1]) + int(areas[1])+30)

            # cv2.line(frame, p1, p2, (255,0,0), 2)
            cv2.rectangle(frame, (int(areas[0]), int(areas[1])), (int(areas[2]), int(areas[3])), (0, 255, 0), 1)
            for i in range(len(five_landmark) // 2):
                 landmark = (int(five_landmark[i]), int(five_landmark[len(five_landmark) -1 -i]))
                 cv2.circle(frame, landmark, 1, (0, 0, 255), 1)
            # landmark = landmark.tolist()
            # for i in range(len(landmark)):
            #     l = landmark[i]s
            #     landmark[i] = (int(l[0]), int(l[1]))
            #     cv2.circle(frame, landmark[i], 1, (0, 0, 255), 1)
        # cv2.imwrite(r"E:\papers\face\test1220\test_lv/{}.jpg".format(nums), frame)
        cv2.imshow('frame',frame)
        input()
        cv2.imwrite("./save_imgs/{}.jpg".format(nums), frame)
        nums += 1
        if cv2.waitKey(1) &0xFF == ord('q'):
            break

cv2_cam()