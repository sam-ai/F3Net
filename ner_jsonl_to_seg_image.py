import jsonlines
import os
import cv2
from tqdm import tqdm
import numpy as np


def change_selected_segmentation_image(
        jsonl_file_path,
        folder_path,
        store_path,
        choosed_label
):
    with jsonlines.open(jsonl_file_path) as reader:
        ner_list = list(reader)

    for obj in ner_list:
        if obj['answer'] == "accept":
            filename = obj['filename']
            data_img_path = os.path.join(folder_path, filename)
            img = cv2.imread(data_img_path)
            height = img.shape[0]
            width = img.shape[1]
            blank_image = np.zeros((height, width, 3), np.uint8)

            selected_label = [
                obj['spans'][index]
                for index in list(obj['spans'].keys())
                if obj['spans'][index]['label'] in choosed_label
            ]
        for bounding_box in selected_label:
            x1, y1, x2, y2 = bounding_box['region']['x1'], \
                             bounding_box['region']['y1'], \
                             bounding_box['region']['x2'], \
                             bounding_box['region']['y2']
            cv2.rectangle(
                blank_image,
                (x1, height - y1),
                (x2, height - y2),
                (255, 255, 255),
                cv2.FILLED
            )
            cv2.imwrite(os.path.join(store_path, filename), blank_image)


if __name__=='__main__':
    choosed_label = [
        "description_of_item",
        "header_of_categories",
        "item_amount",
        "GST_amount",
        "grand_total"
    ]

    change_selected_segmentation_image(
        "/content/kpmg_ner.jsonl",
        "/content/drive/My Drive/KPMG_datasets/chargrid_image",
        "/content/segment_image",
        choosed_label
    )