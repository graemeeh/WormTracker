import numpy as np
import cv2
import pandas as pd
import scipy

def batch_track(folder):
    pass

class Track(object):
    def __init__(self, backgroundsubtractor, kernel_size, stdev, ):
        pass

def bg_subtract(img, bgsubtractor, ksize: int = 5):
    """"
    see https://doi.org/10.1016/j.patrec.2005.11.005 for how this works with MOG2
    Subtracts background using input background subtraction method, initialized outside of the function.
    :param img: input image
    :param bgsubtractor: input background subtraction method
    :param ksize: kernel size for https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    :return: image
    """
    return cv2.morphologyEx(bgsubtractor.apply(img), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))


def log_abs(img, sigma: float = 0.2):
    """
    see https://www.youtube.com/watch?v=uNP6ZwQ3r6A for how this works
    :param img:
    :param sigma:
    :param thresh:
    :return:
    """
    img = cv2.Laplacian(cv2.GaussianBlur(img, (0, 0), sigma), cv2.CV_64F) # note data type cv_64f
    return cv2.normalize(np.abs(img), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # gets absolute value & scales the pixel values of array so that theyare between 0-255

def worms_track(edges, min: int, max: int, ratio: float, time: int):
    """
    
    :param edges: input image
    :param min: minimum area
    :param max: max area
    :param ratio: ratio between circumference and area
    :param time: frame index
    :return: pandas dataframe, highlighted image
    """
    labeled_image, num_labels = scipy.ndimage.label(edges)
    slices = scipy.ndimage.find_objects(labeled_image) # chops it up into slices based on labels
    contour_data = []
    highlighted = np.zeros_like(edges, dtype=np.uint8)
    for i, slice in enumerate(slices, 1):  # start from 1 because 0 is background
        contour_mask = (labeled_image[slice] == i)
        filled = np.zeros_like(contour_mask, dtype=np.uint8)
        peri = np.sum(contour_mask)
        scipy.ndimage.binary_fill_holes(contour_mask, structure=np.ones((3, 3)), output=filled)
        area = np.sum(filled)
        if min < area < max and peri/area < ratio:
            cx, cy = np.mean(np.column_stack(np.where(contour_mask)), axis=0)
            highlighted[slice][contour_mask] = 255
            contour_data.append({
                'index': i,
                'frame': time,
                'centroid_x': int(cx + slice[1].start),
                'centroid_y': int(cy + slice[0].start),
                'area': area,
                'convex_hull_area': int(scipy.spatial.ConvexHull(np.column_stack(np.where(contour_mask))).volume),
                'circumference': peri,
                'height': slice[0].stop - slice[0].start,
                'width': slice[1].stop - slice[1].start})
    return pd.DataFrame(contour_data), highlighted

class Features(object):
    def __init__(self):
        pass

def median_frame(vid):
    # This is from https://github.com/samwestby/OpenCV-Python-Tutorial/blob/main/8_background_est.py
    frame_ids = [i for i in range(1, 41)]
    frames = []
    for f in frame_ids:
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        r, f = vid.read()
        if not r:
            print("SOMETHING WENT WRONG!!!!")
            exit()
        frames.append(f)
    return cv2.cvtColor(np.median(frames, axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY)


def feats_detect(vid, min: int = 10000, max: int = 10000000, ksize: int = 7):
    """

    :param min:
    :param max:
    :param vid:
    :param ksize:
    :return:
    """
    # below bit is from me
    morph = cv2.morphologyEx(median_frame(vid), cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))
    blur_gray = cv2.adaptiveThreshold(morph,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,35,2)
    blur_gray3 = worms_track(log_abs(img = blur_gray, sigma=0.5), min = min, max = max, ratio = 1, time = 1)[1]
    line_image = cv2.cvtColor(np.copy(blur_gray3) * 0, cv2.COLOR_GRAY2BGR)
    # below is from cv2 docs
    details = []
    '''
    lines = cv2.HoughLinesP(blur_gray3, rho = 1, theta = np.pi / 180, threshold = 500, lines = np.array([]), minLineLength = 500, maxLineGap = 20 )
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            details.append({'type': 'line','x1': x1,'y1': y1,'x2/radius': x2,'y2': y2})
    '''
    circles = cv2.HoughCircles(blur_gray3, cv2.HOUGH_GRADIENT, 1.5, blur_gray3.shape[0]/20, param1=200, param2=600, minRadius=100, maxRadius= int(blur_gray3.shape[0]/1.5))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(line_image, (i[0], i[1]), 3, color = (255, 255, 0), thickness=5)
            cv2.circle(line_image, (i[0], i[1]), i[2], color = (255, 0, 255), thickness=5)
            details.append({'type': 'circle', 'x1': i[0], 'y1': i[1], 'x2/radius': i[2], 'y2': 0})
    return pd.DataFrame(details), cv2.addWeighted(cv2.cvtColor(median_frame(vid), cv2.COLOR_GRAY2BGR), 0.5, line_image, beta =0.9, gamma = 0)

def join_tracks(f: pd.DataFrame, maxdist: int):
    """
    :param f: data frame with columns for index, frame (aka time point), centroid_x, centroid_y, area, convex_hull_area (all of which define a track)
    :param maxdist: maximum distance over which tracks can be joined. tracks further away cannot be joined.
    :return: new pandas dataframe. has tracks joined (joined tracks have same index, different frames)
    """
    f['index'] = [i for i in range(0, len(f))]
    for i in pd.unique(f['frame']):
        sub = f[f["frame"] == i]
        x = sub['centroid_x'].values
        y = sub['centroid_y'].values
        # optimize distance between tracks in this frame and tracks in the following frame. Join tracks by updating indices in the following frame to match the joined tracks.
        # maybe also optimize area difference between subsequent frames? cost matrix from https://github.com/Tierpsy/tierpsy-tracker/blob/development/tierpsy/analysis/traj_join/joinBlobsTrajectories.py
    return f



FILENAME = "N2.avi"
VIDEO = cv2.VideoCapture(FILENAME)
mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 1024, 768)

ret = True
time = 0
con_d = []
paused = False

while ret:
    ret, frame = VIDEO.read()
    if frame is None:
        break
    frame = cv2.ximgproc.anisotropicDiffusion(frame, alpha = 0.2, K = 100, niters= 3)
    edges = log_abs(bg_subtract(img = frame, bgsubtractor=mog), sigma = 0.3)
    contour_d, highlighted_image = worms_track(edges, min=150, max=1500, time = time, ratio = 0.8)
    f = cv2.addWeighted(frame, 0.5, cv2.merge([highlighted_image, np.zeros_like(highlighted_image), highlighted_image]), 0.9, 0)
    resized_frame = cv2.resize(f, (1024, 768))
    cv2.imshow('Video', resized_frame)
    time += 1
    con_d.append(contour_d)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    while paused:
        key = cv2.waitKey(100) & 0xFF
        if key == ord(' '):
            paused = False
            break

d, b = feats_detect(VIDEO)
while True:
    cv2.imshow('Video', b)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

VIDEO.release()
cv2.destroyAllWindows()


df = pd.concat(con_d)
join_tracks(df, 500)
df.to_csv(FILENAME+'_worms.csv')
d.to_csv(FILENAME+'_details.csv')
print("all done!")
