import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def detect_and_count(save_txt=False, save_img=False, ROI="vertical"):
    # (320, 192) or (416, 256) or (608, 352) for (height, width)
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(
        device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(
            name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        # set True to speed up constant image size inference
        torch.backends.cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(classes))]

    # Cumulative trackers in the frame
    tracks_active = []
    tracks_finished = []

    cumulative_count = 0

    total_L2R = 0
    total_R2L = 0
    total_U2D = 0
    total_D2U = 0

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Apply
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            persons = []     # store only persons detected
            updated_tracks = []        # updated tracks for every frame

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            H, W = im0.shape[0:2]

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        # filter if object larger than half of the frame area
                        if classes[int(cls)] == "person" and (xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1]) < H*W * 0.6:
                            label = '%s %.2f' % (classes[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label,
                                         color=colors[int(cls)])

                # Append only people detected
                for de in det:
                    # filter if object larger than half of the frame area
                    if classes[int(de[-1])] == "person" and (de[2]-de[0]) * (de[3]-de[1]) < H*W * 0.6:
                        de = list(de[0:5].cpu().numpy())
                        person = {
                            "bbox": de[0:4],
                            "score": de[4]
                        }
                        persons.append(person)

            for track in tracks_active:
                if len(persons) > 0:
                    best_match = max(persons, key=lambda x: iou(
                        track['bboxes'][-1], x['bbox']))
                    # default sigma_iou
                    if iou(track['bboxes'][-1], best_match['bbox']) >= 0.5:
                        track['bboxes'].append(best_match['bbox'])
                        track['max_score'] = max(
                            track['max_score'], best_match['score'])

                        updated_tracks.append(track)

                        del persons[persons.index(best_match)]

                    if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                        # default sigma_h and t_min
                        if track['max_score'] >= 0.5 and len(track['bboxes']) >= 2:
                            tracks_finished.append(track)

            # create new tracks
            new_tracks = [{
                'bboxes': [person['bbox']],
                'max_score': person['score'],
                'centroid': (int((person['bbox'][2] + person['bbox'][0])/2), int((person['bbox'][3] + person['bbox'][1])/2)),
                'direction': None,
                'counted': False
            } for person in persons]

            for track in new_tracks:
                if ROI == "vertical":
                    if track['centroid'][0] < W//2:
                        track['direction'] = "L2R"
                    else:
                        track['direction'] = "R2L"
                elif ROI == "horizontal":
                    if track['centroid'][1] < H//2:
                        track['direction'] = "U2D"
                    else:
                        track['direction'] = "D2U"

            tracks_active = updated_tracks + new_tracks

            for track in tracks_active:
                centroid = (int((track["bboxes"][-1][2] + track["bboxes"][-1][0])/2),
                            int((track["bboxes"][-1][3] + track["bboxes"][-1][1])/2))
                track['centroid'] = centroid
                im0 = cv2.circle(im0, centroid, 10, (0, 255, 0), -1)
                # print(track)  

                if track['direction'] == "L2R" and track['centroid'][0] > W//2 and track['counted'] == False:
                    total_L2R += 1
                    track['counted'] = True

                if track['direction'] == "R2L" and track['centroid'][0] < W//2 and track['counted'] == False:
                    total_R2L += 1
                    track['counted'] = True

                if track['direction'] == "U2D" and track['centroid'][1] > H//2 -100 and track['counted'] == False:
                    total_U2D += 1
                    track['counted'] = True

                if track['direction'] == "D2U" and track['centroid'][1] < H//2 -100 and track['counted'] == False:
                    total_D2U += 1
                    track['counted'] = True

            # finish all remaining active tracks
            tracks_finished += [track for track in tracks_active if track['max_score']
                                >= 0.5 and len(track['bboxes']) >= 2]

            if ROI == 'vertical':
                text = 'Detected People: ' + str(len(tracks_active)) + ",  total Left -> Right: " + str(
                    total_L2R) + ",  total Right -> Left: " + str(total_R2L)
            elif ROI == 'horizontal':
                text = 'Detected People: ' + str(len(tracks_active)) + ",  total Up -> Down: " + str(
                    total_U2D) + ",  total Down -> Up: " + str(total_D2U)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                im0,
                text,
                (10, H - 20),
                font,
                0.8,
                (0, 0xFF, 0xFF),
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )

            if ROI == 'horizontal':
                cv2.line(im0, (0, H // 2 - 100), (W, H // 2 - 100), (0, 0, 255), 2)
            elif ROI == 'vertical':
                cv2.line(im0, (W // 2, 0), (W // 2, H), (0, 0, 255), 2)

            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str,
                        default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str,
                        default='weights/yolov3-spp.weights', help='path to weights file')
    # input file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='data/samples', help='source')
    parser.add_argument('--output', type=str, default='output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true',
                        help='half precision FP16 inference')
    parser.add_argument('--device', default='',
                        help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect_and_count(ROI='horizontal')
