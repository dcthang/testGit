try:
    from PIL import Image
except ImportError:
    import Image
import cv2
import numpy as np
# import numpy as np
# import math
import pytesseract
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from hmer import Hmer
from onmt.translate.translation_im import TranslationImCli

#pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

custom_config = r'-l vie'  # --psm 1,3,4, # --oem 3 # vie+eng+equ # --psm 1 --oem 1
CV_PI = 3.1415926535897932384626433832795

hmer = None
onmt = None
predictor = None

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)


def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return 0 # or (0,0,0,0) ?
    return (w * h)


def detect_text(img):
    # img = Image.open(filename)
    # text = pytesseract.image_to_string(img, config=custom_config)

    lines = []
    line_idx = 0

    data=pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DATAFRAME)
    data = data[data["conf"] > 0]
    # print(data)
    data_group = data.groupby(['block_num', 'line_num']);

    # print(data_group['left'].apply(list))

    # print(data_group.groups)
    lines_bbox = []

    for name, group in data_group:
        line = []
        bbox = None

        for index, row in group.iterrows():
            # print(row['conf'], row['text'])
            line.append({
                "text": row["text"],
                "box": [row["left"], row["top"], row["width"], row["height"]],
                "block_num": str(row["block_num"]) + "_" + str(row["line_num"])
            })
            # bbox
            # if bbox is not None:
            #     bbox = union(bbox, [row["left"],row["top"],row["width"],row["height"]])
            # else:
            bbox = [row["left"],row["top"],row["width"],row["height"]]
        # print("----------------")
        lines.append(line)
        lines_bbox.append(bbox)

    # lines = text.groupby('block_num')['text'].apply(list)
    # print(len(lines))
    # image = cv2.imread(filename, cv2.IMREAD_COLOR)
    
    color_idx = 0
    color_table= [(0,0,0), (0,0,255), (0,255,0), (0,255,255), (255,0,0), (255,0,255), (255,255,0)]
    for blocks in lines:
        for item in blocks:
            cv2.rectangle(img, (item["box"][0], item["box"][1]), (item["box"][0]+ item["box"][2], item["box"][1]+ item["box"][3]),color_table[color_idx])
        color_idx += 1
        if color_idx >= len(color_table):
            color_idx = 0

    for bbox in lines_bbox:
        if bbox is not None:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+ bbox[2], bbox[1]+ bbox[3]),color_table[5])
    # cv2.imshow("image", img)

    return lines, lines_bbox


def detect_fomular(im, model='onmt'):
    results = []

    global hmer, onmt, predictor

    im = cv2.imread(im)

    if predictor is None:
        cfg = get_cfg()

        cfg.merge_from_file("configs/mfr_mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        #cfg.merge_from_file("configs/mask_rcnn_R_50_C4_3x.yaml")

        cfg.MODEL.WEIGHTS = "./model/model_final.pth"
        
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model 0.6
#        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3  # # Overlap threshold used for non-maximum suppression (suppress boxes with 0.4
        # IoU >= this threshold)
        cfg.DATASETS.TEST = ("mfr_test", )
        predictor = DefaultPredictor(cfg)

    if model == 'hmer' and hmer is None:
        hmer = Hmer()
    elif model == 'onmt' and onmt is None:
        onmt = TranslationImCli()
        onmt.start('./configs/trans.conf.json')
    
    outputs = predictor(im)
    instances = outputs["instances"]
    
    pred_boxes = []
    for i, box in enumerate(instances.get("pred_boxes")):
        box = box.int()
#        print("box", i, box)
        is_intersect = False
        # print(" ")
        #boxes.pairwise_intersection()
        # for ii in range(len(pred_boxes)) :
        #     if intersection(pred_boxes[ii], box[0:4]) > 0:
        #         xmin,ymin,xmax,ymax = box[0:4]
        #         is_intersect = True
        #         if pred_boxes[ii][0] > xmin:
        #             pred_boxes[ii][0] = xmin
        #         if pred_boxes[ii][1] > ymin:
        #             pred_boxes[ii][1] = ymin
        #         if pred_boxes[ii][2] < xmax:
        #             pred_boxes[ii][2] = xmax
        #         if pred_boxes[ii][3] < ymax:
        #             pred_boxes[ii][3] = ymax

#        if not is_intersect:
        pred_boxes.append(box[0:4])

    for i, box in enumerate(pred_boxes):
        xmin,ymin,xmax,ymax = box[0:4]

        cropped_img = im[ymin.item(): ymax.item(), xmin.item(): xmax.item()]
        latex = "N/A"
        if hmer is not None:
            latex = hmer.predict(cropped_img)
        elif onmt is not None:
            img_binary = Hmer.convert_to_binary(cropped_img)
            image = Image.fromarray(img_binary)
            _rs, scores, n_best, times = onmt.run([{"src": image}])
            latex = _rs[0]
        # show debugging
        # h, w = cropped_img.shape[:2]
        # img_text = np.zeros((100,w,3), np.uint8)
        # cv2.putText(img_text,latex,(0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)

        # img_threshold = Hmer.convert_to_binary(cropped_img)
        # img_tmp = np.zeros_like(cropped_img)
        # img_tmp[:,:,0] = img_threshold # same value in each channel
        # img_tmp[:,:,1] = img_threshold
        # img_tmp[:,:,2] = img_threshold
        # print("img_text", img_text.shape, " crop", cropped_img.shape, " img_threshold", img_threshold.shape, "img_tmp", img_tmp.shape)

        # im_v = cv2.vconcat([cropped_img, img_tmp ,img_text])

        # cv2.imshow("final "+str(i), im_v)
#        print("latex, ", latex)
        # end debugging

        results.append({"box":[xmin.item(),ymin.item(),xmax.item()-xmin.item(),ymax.item()-ymin.item()], "text": latex})

    from detectron2.utils.visualizer import ColorMode

    MetadataCatalog.get("mfr_test").set(thing_classes=["formula"])
    metadata_val = MetadataCatalog.get("mfr_test") # for d in random.sample(dataset_dicts, 3):

    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata_val,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    import matplotlib.pyplot as plt
    plt.imshow(v.get_image()[:, :, ::-1], cmap='gray')
    plt.show()

    #cv2.imshow("x",v.get_image()[:, :, ::-1])
    # cv2.waitKey();
    return results


def detect_fomular_x(im, model='hmer'):
    """
    for dectect formula only

    """
    latex = ""
    global hmer, onmt

    if model == 'hmer' and hmer is None:
        hmer = Hmer()
    elif model == 'onmt' and onmt is None:
        onmt = TranslationImCli()
        onmt.start('./configs/trans.conf.json')


    if hmer is not None:
        latex = hmer.predict(im)
    elif onmt is not None:
        img_binary = Hmer.convert_to_binary(im)
        image = Image.fromarray(img_binary)
        _rs, scores, n_best, times = onmt.run([{"src": image}])
        latex = _rs[0]
    # show debugging

    return latex


def predict_final(filename):

    im = cv2.imread(filename)

    #formulas = detect_fomular(im, model='hmer')
    
    formulas = detect_fomular(im, model='onmt')
    #formulas = detect_fomular_x(im, model='onmt')
    
    lines, lines_bbox = detect_text(im)

#    print("lines_bbox", lines_bbox)
    # sap xep formula vao tung line
    # todo: truong hop formula ko nam trong line
    for i in range(len(formulas)):
        max_area = 0
        line_idx = 0
        for j in range(len(lines_bbox)):
            if lines_bbox[j] is None:
                continue
            tmp_area = intersection(formulas[i]['box'], lines_bbox[j])
            if tmp_area > max_area:
                max_area = tmp_area
                line_idx = j
        formulas[i]['line_idx'] = line_idx
        # if "formulas" not in lines_bbox[line_idx]:
        #     lines_bbox[line_idx]["formulas"] = []
        # lines_bbox[line_idx]['formulas'].append(formulas[i])

    #print("formula", formulas)

    #tra ket qua theo tung dong tung
    results = []
    for i in range(len(lines)):
        res = ""

        for text in lines[i]:
            # kiem tra xem item co nam trong formulas hay ko
            ok_to_add = True

            for f_idx, formula in enumerate(formulas):
                tmp_area = intersection(formula["box"], text["box"])
                if tmp_area > 0:
                    ok_to_add = False

                if "line_idx" not in formula or formula["line_idx"] != i:
                    continue

                # den luc insert formulas vao hay chua ??
                if formula["box"][0] < text["box"][0] and "pass" not in formula:
                    # print("debug, ",formula, "i ", i)
                    res += "$$" + formula["text"] + "$$ "
                    formula["pass"] = True
                    # print(formula)
                    # print(text)

            if ok_to_add:
                res += text["text"] + " "

        # print("now formulas", formulas)
        # kiem tra formulas lan cuoi xem co sot ko
        for f_idx, formula in enumerate(formulas):
            if "line_idx" not in formula or formula["line_idx"] != i or "pass" in formula:
                continue
            res += "$$" + formula["text"] + "$$ "
            formulas[f_idx]["pass"] = True
        # todo: truong hop formula ko thuoc line nao
        results.append(res)

    return "\n".join(results)
    # cv2.waitKey()


#/Users/thanhtruongle/Downloads/20200611_Sample-test/a50/image248.jpg
if __name__ == '__main__':
    #print(predict_final("/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image124.jpg"))
    print(detect_fomular("/home/dcthang/Projects/MathFormulaReg/Docs/Test/20200724_datatest/A50/image083.jpg", "onmt"))
    #print(predict_final("/home/dcthang/Projects/MathFormulaReg/Code/mfr-pytorch-hmer/data/csv3/images4/toan_12_page_149_6.jpg"))


