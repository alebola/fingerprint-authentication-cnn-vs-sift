import os, cv2, numpy as np, random


def recortar_imagen(input_folder, output_folder, crop_width=550, crop_height=550, threshold_value=225):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if not file.lower().endswith(('.png','.jpg','.jpeg')): continue
            p = os.path.join(root, file)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            _, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            cx, cy = x+w//2, y+h//2
            xs, ys = max(cx-crop_width//2,0), max(cy-crop_height//2,0)
            xe, ye = min(xs+crop_width, img.shape[1]), min(ys+crop_height, img.shape[0])
            if (xe-xs)!=crop_width: xs = max(0, xe-crop_width)
            if (ye-ys)!=crop_height: ys = max(0, ye-crop_height)
            cropped = img[ys:ye, xs:xe]
            out = os.path.join(output_folder, file)
            cv2.imwrite(out, cropped)


def procesar_imagen(image_path, output_path,
                    clahe_clip=1.0, clahe_grid=(8,8),
                    bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
                    median_ksize=3,
                    adapt_block=19, adapt_C=3,
                    morph_kernel=3,
                    use_skeleton=False,
                    final_size=(550,550)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    eq = clahe.apply(image)
    bil = cv2.bilateralFilter(eq, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)
    med = cv2.medianBlur(bil, median_ksize)
    th = cv2.adaptiveThreshold(med, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adapt_block, adapt_C)
    inv = cv2.bitwise_not(th)
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    clean = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    out = clean if not use_skeleton else _skeletonize(clean)
    out = cv2.resize(out, final_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, out)


def _skeletonize(binary_img):
    skel = np.zeros(binary_img.shape, np.uint8)
    img = binary_img.copy()
    while True:
        eroded = cv2.erode(img, None)
        temp = cv2.dilate(eroded, None)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def new_images(input_dir, num_images=10, cut_on_rejected=False):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    for name in files:
        p = os.path.join(input_dir, name)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        h, w = img.shape
        for i in range(num_images):
            angle = random.randint(-90, 90)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            aug = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            sx, sy = random.uniform(0.8,1.2), random.uniform(0.8,1.2)
            aug = cv2.resize(aug, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)
            aug = cv2.resize(aug, (w,h), interpolation=cv2.INTER_LINEAR)
            if cut_on_rejected and input_dir.endswith('rejected'):
                x1, y1 = random.randint(0, w//8), random.randint(0, h//8)
                x2, y2 = random.randint(x1, w//4), random.randint(y1, h//4)
                cv2.rectangle(aug, (x1,y1), (x2,y2), (0,0,0), -1)
            tx, ty = random.randint(-10,10), random.randint(-10,10)
            T = np.float32([[1,0,tx],[0,1,ty]])
            aug = cv2.warpAffine(aug, T, (w,h), borderMode=cv2.BORDER_REFLECT_101)
            cv2.imwrite(os.path.join(input_dir, f"{i}_{name}"), aug)