# import libraries here
import matplotlib
import matplotlib.pyplot as plt
import numpy as bb8
import cv2

def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca, belih krvnih zrnaca i
    informaciju da li pacijent ima leukemiju ili ne, na osnovu odnosa broja krvnih zrnaca

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca,
             <int> broj prebrojanih belih krvnih zrnaca,
             <bool> da li pacijent ima leukemniju (True ili False)
    """
    # TODO - Prebrojati crvena i bela krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure

    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure

    white_blood_cell_count, not_important, wbc_area = count_wbc(image_path)
    red_blood_cell_count, rbc_area = count_red_blood_cells(image_path)

    data = image_path.split(';')
    picture = data[0].split('-')
    number = picture[1].split('.')

    has_leukemia = patient_has_leukemia(white_blood_cell_count, red_blood_cell_count)
    print("number: ", number, red_blood_cell_count, white_blood_cell_count,has_leukemia)
    print("     WBC area: ", wbc_area," RBC area: ",rbc_area)
    print("     Ratio: ", float(wbc_area/rbc_area), " Leukemia? : ", has_leukemia)
    return red_blood_cell_count, white_blood_cell_count, has_leukemia

# In[1]:
def patient_has_leukemia(wbc_cnt, rbc_cnt):
    has_leukemia = None
    try:
        var = float(wbc_cnt / rbc_cnt)
        has_leukemia = (False, True)[var > 0.09]
    except:
        has_leukemia = True

    return has_leukemia
# In[2]:
def split2hsv(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img)
    return h, s, v
# In[3]:
def remove_inner_contour_rbc(img, contours, hierarchy):
    i = 0
    new_contours = []
    contour_area_avg = 0

    for contour in contours:
        if hierarchy[0, i, 3] == -1 and cv2.contourArea(contour)>100000.0:
            new_contours.append(contour)
        i = i + 1

    cv2.drawContours(img, new_contours, -1, (255, 0, 0), 1)
    return img, new_contours
# In[4]:
def do_floodfill(img_bin):
    height, width = img_bin.shape[:2]
    img_for_floodfill = img_bin.copy()

    mask = bb8.zeros((height+2, width+2), bb8.uint8)
    cv2.floodFill(img_for_floodfill, mask, (0,0), 255)
    floodfill_inverted = cv2.bitwise_not(img_for_floodfill)
    img_floodfilled = img_bin | floodfill_inverted
    return img_floodfilled
# In[5]:
def draw_contour(img_bin):
    img_cont, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_cont, contours, -1, (255, 0, 0), 1)
    return img_cont, contours, hierarchy


# In[6]:
def remove_inner_and_small_contours(img, contours, hierarchy):
    i = 0
    new_contours = []
    cells_area = 0.0
    for contour in contours:
        if hierarchy[0, i, 3] == -1 and cv2.contourArea(contour) > 160.0:
            new_contours.append(contour)
            cells_area = cells_area + cv2.contourArea(contour)
        i = i + 1

    cv2.drawContours(img, new_contours, -1, (255, 0, 0), 1)
    return img, new_contours, cells_area


# In[7]:
def brightness_up(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - 30
    v[v > lim] = 255
    v[v <= lim] += 30

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img

# In[8]:
def brightness_down(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - 30
    v[v > lim] = 255
    v[v <= lim] -= 30

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img
# In[9]:
def smoothen_img(img):
    kernel_smooth = bb8.ones((5,5),bb8.float32)/25
    filtered2d = cv2.filter2D(img,-1,kernel_smooth)
    #plt.imshow(filter2d)
    return filtered2d


# In[10]:
def contrastLab(img_in_bgr):
    img_lab= cv2.cvtColor(img_in_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe_stuff = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe_stuff.apply(l)

    limg = cv2.merge((cl,a,b))


    contrasted_lab_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return contrasted_lab_img

# In[11]:
def prepare_rbc_img_for_threshold(img_lab):
    img_contrast = contrastLab(img_lab)
    img_hsv = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_minus = 0 - img_gs

    return img_minus

# In[12]:
def count_red_blood_cells(img_path):
    img = cv2.imread(img_path)
    img_copy = img.copy()
    #img = brightness_down(img)
    #img_smoothen = smoothen_img(img)
    img_contrast = contrastLab(img)
    # plt.imshow(img_contrast)
    img_hsv = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_t = 0 - img_gs
    ret, thresh = cv2.threshold(img_t, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_bin = 255 - thresh

    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    dilation = cv2.erode(thresh, kernel_ellipse, iterations=1)
    dilation = cv2.erode(dilation, kernel_cross, iterations=1)
    white_mask = count_wbc(img_path)[1]
    dd = white_mask + dilation
    erode_dd = cv2.dilate(dd, kernel_ellipse, iterations=2)
    erode_dd = cv2.dilate(erode_dd, kernel_cross, iterations=1)
    closing = cv2.morphologyEx(erode_dd, cv2.MORPH_OPEN, kernel_ellipse, iterations=3)
    erode_dd = cv2.erode(closing, kernel_cross, iterations=2)

    erode_dd = 255 - erode_dd
    img_cont, contours, hierarchy = draw_contour(erode_dd)
    img_contoured_final, new_contours, rbc_area = remove_inner_and_small_contours(img_copy, contours, hierarchy)
    red = (len(new_contours))
    plt.imshow(img_contoured_final)
    plt.show()
    return red,rbc_area

# In[13]:
def count_wbc(img_path):
    img = cv2.imread(img_path)
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, s, v = split2hsv(img)
    # plt.imshow(s,'gray')
    ret, img_bin = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)
    # plt.imshow(image_bin,'gray')
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    img_bin_opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_ellipse)
    # plt.imshow(img_bin_opening,'gray')
    img_bin_eroded= cv2.erode(img_bin_opening, kernel_ellipse, iterations=2)
    img_bin_eroded = cv2.erode(img_bin_eroded, kernel_cross, iterations=1)

    ff = do_floodfill(img_bin_eroded)
    # plt.imshow(ff,'gray')
    # img_bin_erode_ff = cv2.erode(ff,bb8.ones((4,4),bb8.uint8),iterations = 3)
    dist_transform = cv2.distanceTransform(ff, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    sure_fg = bb8.uint8(sure_fg)
    plt.imshow(sure_fg)
    plt.show()
    img_cont, contours, hierarchy = draw_contour(sure_fg)
    img_contoured_final, new_contours, wbc_area = remove_inner_and_small_contours(img_copy, contours, hierarchy)
    #print(len(new_contours))
    wbc_cnt = len(new_contours)
    #plt.imshow(img_contoured_final)
    return wbc_cnt,img_bin,wbc_area
