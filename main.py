import math
import os
import shutil
import cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.cluster import KMeans
from routers import ToolRoutes
# uvicorn main:app --reload คำสั่งรัน API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_path = os.path.abspath(os.getcwd())
faceCascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier(current_path+'\haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier(current_path+'\haarcascade_mouth.xml')

master_colors = [
    (249, 222, 203), (241, 214, 193), (254, 212,
                                       196), (248, 217, 189), (238, 200, 179),
    (248, 202, 189), (248, 209, 176), (228, 191,
                                       165), (219, 182, 155), (238, 192, 176),
    (226, 182, 143), (218, 174, 149), (206, 164,
                                       139), (225, 174, 153), (227, 182, 149),
    (222, 184, 137), (203, 159, 132), (216, 161,
                                       140), (223, 168, 128), (196, 143, 101),
    (198, 152, 119), (202, 141, 122), (178,
                                       124, 86), (196, 142, 98), (188, 144, 117),
    (180, 121, 87), (184, 140, 111), (174,
                                      125, 85), (171, 130, 102), (157, 108, 78),
    (160, 100, 63), (152, 101, 71), (176, 114, 93), (159, 99, 63), (127, 81, 58),
    (161, 100, 82), (138, 83, 53), (114, 69, 50), (140, 81, 67), (89, 56, 47),
    (45, 34, 30), (60, 46, 40), (75, 57, 50), (90, 69, 60), (105, 80, 70),
    (120, 92, 80), (135, 103, 90), (150, 114,
                                    100), (165, 126, 110), (180, 138, 120),
    (195, 149, 130), (210, 161, 140), (225, 172,
                                       150),  (240, 184, 160), (255, 195, 170),
    (255, 206, 180), (255, 218, 190), (255, 229,
                                       200), (219, 162, 135), (242, 215, 190),
    (246, 218, 193), (238, 178, 155), (248,
                                       196, 174), (254, 212, 195), (191, 90, 75),
    (220, 161, 133), (224, 166, 140), (228, 151,
                                       103), (244, 177, 141), (254, 207, 177),
    (239, 172, 124), (234, 175, 131), (241, 199,
                                       163), (224, 165, 124), (237, 186, 144),
    (222, 166, 124), (195, 119, 71), (228, 147, 97), (249, 195, 164), (192, 90, 47),
    (187, 97, 50), (211, 135, 99), (188, 102, 42), (177, 94, 44), (213, 114, 44),
    (182, 103, 64), (194, 112, 71), (212, 145,
                                     94), (224, 166, 122), (246, 200, 172),
    (248, 220, 200), (207, 133, 97), (236, 185,
                                      145), (240, 192, 167), (190, 130, 86),
    (222, 165, 117), (226, 180, 141), (185,
                                       121, 78), (192, 133, 89), (207, 151, 108),
    (161, 93, 50), (190, 120, 59), (210, 171,
                                    143), (228, 164, 104), (239, 187, 127),
    (247, 201, 162), (183, 119, 69), (202, 137, 93), (212, 153, 98), (179, 101, 51),
    (193, 117, 65), (204, 139, 78), (186, 127, 83), (174, 106, 49), (216, 146, 91),
    (217, 147, 86), (225, 163, 104), (234, 177,
                                      119), (216, 170, 131), (239, 191, 150),
    (245, 200, 171), (221, 161, 102), (231,
                                       205, 205), (241, 181, 132), (235, 161, 97),
    (230, 179, 109), (244, 200, 158), (192,
                                       141, 90), (170, 103, 57), (183, 124, 76),
    (199, 124, 66), (211, 141, 69), (230, 158, 96), (175, 115, 65), (202, 128, 65),
    (226, 162, 95), (232, 159, 93), (233, 167,
                                     116), (245, 190, 149), (188, 111, 55),
    (198, 123, 66), (214, 139, 80), (137, 70, 36), (163, 83, 42), (176, 95, 48),
    (189, 126, 83), (184, 100, 46), (138, 75, 41), (215, 187, 198), (219, 155, 90),
    (223, 170, 115), (185, 120, 63), (202, 149, 86), (207, 123, 56), (116, 76, 41),
    (177, 118, 56), (189, 113, 44), (79, 42, 19), (198, 119, 60), (154, 90, 31),
    (160, 88, 23), (184, 108, 43), (207, 156, 92), (90, 53, 22), (121, 68, 29),
    (139, 77, 25), (94, 46, 19), (131, 57, 30), (157, 81, 57), (79, 36, 16),
    (111, 53, 31), (129, 72, 48), (69, 31, 15), (97, 44, 21), (73, 31, 6),
    (60, 29, 10), (43, 23, 7), (46, 20, 7), (185, 119, 54), (194, 133, 72),
    (224, 151, 98), (211, 142, 67), (179, 113, 50), (195, 140, 72), (163, 102, 80),
    (185, 118, 70), (216, 149, 87), (119, 79, 47), (164, 102, 59), (197, 132, 83),
    (123, 68, 43), (140, 82, 47), (183, 127, 82), (122, 83, 60), (124, 74, 50),
    (149, 95, 62), (97, 58, 30), (126, 82, 42), (156, 111, 81), (59, 39, 23),
    (68, 42, 22), (102, 60, 36), (45, 27, 10), (72, 41, 19), (87, 50, 26),
    (39, 25, 12), (64, 42, 26), (75, 46, 24), (166, 96, 49), (203, 136, 84),
    (205, 151, 92), (109, 51, 34), (131, 75, 35), (220, 166, 121), (188, 128, 73),
    (147, 83, 46), (180, 116, 65), (148, 86, 37), (173, 117, 51), (207, 143, 91),
    (138, 80, 43), (165, 96, 35), (176, 115, 42), (137, 73, 29), (140, 86, 49),
    (146, 83, 42), (132, 73, 29), (107, 66, 23), (115, 67, 29), (144, 82, 31),
    (76, 46, 22), (73, 49, 29), (151, 99, 64), (56, 32, 15), (58, 37, 17),
    (59, 38, 18), (118, 60, 23), (49, 29, 10), (39, 21, 6), (83, 42, 17),
    (193, 165, 156), (233, 210, 200), (223, 175, 151), (227, 205, 205)

]

master_colors_true = [
    (39, 25, 12), (246, 200, 172), (230, 192, 166), (248, 216, 193), (249, 180, 138),
    (248, 224, 211), (206, 161, 132), (204, 159, 126), (223, 188, 156), (228, 190, 154),
    (228, 185, 140)
]
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, width):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        if width != 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, width)
        coords = [x, y, w, h]
    return coords


def detect(img, faceCascade, eyeCascade, mouthCascade):
    color = {"blue": (255, 0, 0),
             "red": (0, 0, 255),
             "green": (0, 255, 0),
             "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], 0)
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords) == 4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1] +
                      coords[3], coords[0]:coords[0]+coords[2]]
        # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
        coords = draw_boundary(roi_img, eyeCascade, 1.1,
                               12, color['white'], -1)
        coords = draw_boundary(roi_img, mouthCascade, 1.3,
                               25, color['white'], -1)
    return img, roi_img


def visualize_Dominant_colors(cluster, C_centroids):
    C_labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (C_hist, _) = np.histogram(cluster.labels_, bins=C_labels)
    C_hist = C_hist.astype("float")
    C_hist /= C_hist.sum()
    rect_color = np.zeros((50, 300, 3), dtype=np.uint8)
    img_colors = sorted([(percent, color)
                        for (percent, color) in zip(C_hist, C_centroids)])
    return rect_color, img_colors


def rgb_to_hex(r, g, b):
    rn = math.ceil(r)
    gn = math.ceil(g)
    bn = math.ceil(b)
    return ('{:02X}' * 3).format(rn, gn, bn)


def closest_color(r, g, b):
    color_diffs = []
    for color in master_colors_true:
        cr, cg, cb = color
        color_diff = math.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]


@app.get("/")
async def root():
    return {"msg": "Start Prod"}


@app.post("/analysis")
async def root(file: UploadFile = File(...)):
    with open(f'{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    img = cv2.imread(file.filename)
    # Call method we defined above
    img, face = detect(img, faceCascade, eyesCascade, mouthCascade)
    # Writing processed image in a new window
    src_image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    reshape_img = src_image.reshape(
        (src_image.shape[0] * src_image.shape[1], 3))
    KM_cluster = KMeans(n_clusters=5).fit(reshape_img)
    visualize_color, kcolor = visualize_Dominant_colors(
        KM_cluster, KM_cluster.cluster_centers_)
    visualize_color = cv2.cvtColor(visualize_color, cv2.COLOR_RGB2BGR)
    (percent, color) = kcolor[4]
    (r, g, b) = color
    (rr, gg, bb) = closest_color(int(r), int(g), int(b))
    newColor = rgb_to_hex(rr, gg, bb)

    return {"color": newColor}

app.include_router(ToolRoutes.router)