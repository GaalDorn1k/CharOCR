import os
import cv2
import json
import torch
import numpy as np

from tqdm import tqdm
from PIL import Image
from typing import Union, List
from CharOCRBase import CharOCRBase
from DataGenerator import DataGenerator


class CharOCR():
    def __init__(self, device: str, model_config_path: str) -> None:
        with open(model_config_path, 'r', encoding='utf-8') as jf:
            config = json.load(jf)

        model_params = {'alphabet_len': len(config['alphabet']),
                        'fields_num': len(config['fields']),
                        'base_channels': config['base_channels'],
                        'dropout_rate': config['dropout_rate']}
        self.model = CharOCRBase(**model_params)
        self.input_shape = tuple(config['input_shape'])
        if config['weights_path']:
            self.model.load_state_dict(torch.load(config['weight_path']))
        self.field_threshold = config['field_threshold']
        self.device = device
        self.model.to(self.device).double()
        self.alphabet = config['alphabet']
        self.fields = config['fields']

    def __get_contours(self, image: np.ndarray, area_scale: tuple, threshold: tuple) -> List[np.array]:
        thresh = cv2.threshold(image, *threshold, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[0]
        cnts = []
        for contour in contours:
            area = cv2.contourArea(contour)
            rect = cv2.boundingRect(contour)
            image_area = image.shape[0] * image.shape[1]
            if area > area_scale[0] * image_area and area < area_scale[1] * image_area:
                cnts.append((int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])))
        return cnts

    def __get_fields(self, field_mask: np.ndarray, threshold: float) -> dict:
        field_mask = field_mask[0]
        cnts = self.__get_contours(np.uint8(field_mask[0] * 255), area_scale=(0.0005, 0.03), threshold=(100, 255))
        cnts = np.array(cnts)
        # fields = dict.fromkeys(self.fields, [])
        fields = {key: [] for key in self.fields}
        if len(cnts) > 0:
            cnts = cnts[cnts[:, 0].argsort()]
            mask = field_mask[1:].transpose(1, 2, 0)
            for cnt in cnts:
                crop = mask[cnt[1]: cnt[1] + cnt[3], cnt[0]: cnt[0] + cnt[2]]
                for i in range(crop.shape[2]):
                    layer = crop[:, :, i]
                    layer = layer.flatten()
                    probability = sum(layer) / len(layer)
                    if probability < threshold:
                        continue
                    else:
                        fields[self.fields[i]].append(cnt)
        for field_name, rects in fields.items():
            for rect in rects:
                cv2.rectangle(self.img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 5)
                cv2.putText(self.img, field_name, (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # img = Image.fromarray(self.img)
        # img.show()
        return fields

    def __get_texts(self, fields: dict, char_mask: np.ndarray) -> dict:
        # output = dict.fromkeys(list(fields.keys()), {'text': '', 'coord': None})
        output = {key: {'text': '', 'coord': None} for key in list(fields.keys())}
        char_mask = char_mask[0][1:]
        for field_name, coords in fields.items():
            field_text = ''
            try:
                for coord in coords:
                    field_crop = char_mask[:, coord[1]: coord[1] + coord[3], coord[0]: coord[0] + coord[2]]
                    img_crop = self.img2[coord[1]: coord[1] + coord[3], coord[0]: coord[0] + coord[2]]
                    row_chars = {}
                    for index, layer in enumerate(field_crop):
                        cnts = self.__get_contours(np.uint8(layer * 255), area_scale=(0.005, 0.5), threshold=(50, 255))
                        cnts = np.array(cnts)
                        if len(cnts) > 0:
                            if field_crop.shape[1] < field_crop.shape[2]:
                                for cnt in cnts:
                                    row_chars[cnt[0]] = self.alphabet[index]
                            else:
                                for cnt in cnts:
                                    row_chars[cnt[1]] = self.alphabet[index]
                            for rect in cnts:
                                cv2.rectangle(img_crop, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 5)
                    chars_coords = list(row_chars.keys())
                    chars_coords.sort()
                    row = [row_chars[key] for key in chars_coords]
                    past_key = chars_coords[0]
                    delta = []
                    for key in chars_coords:
                        delta.append(key - past_key)
                        past_key = key
                    mean_delta = np.mean(delta)
                    index = None
                    for i, d in enumerate(delta):
                        if abs(d / mean_delta - 1) > 0.3:
                            index = i
                    row_text = ''.join(row)
                    if index:
                        row_text = row_text[:index] + ' ' + row_text[index:]
                    # print(row_text)
                    # thresh = cv2.threshold(field_crop, 0.5, 1, cv2.THRESH_BINARY)[1]
                    # masks = np.array([np.argmax(a, axis=0) for a in thresh])
                    # img = Image.fromarray(img_crop)
                    # img2 = Image.fromarray(masks)
                    # img.show()
                    # img2.show()
                    field_text += row_text + '\n'
            except:
                pass
            # rect = cv2.boundingRect(coords)
            output[field_name]['text'] = field_text
            # output[field_name]['coord'] = rect
        return output

    def predict(self, image: Union[np.ndarray, str]) -> dict:
        self.model.eval()
        self.model.mode = 'eval'
        if isinstance(image, str):
            image = cv2.imread(image)
            image - cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape != self.input_shape:
                image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_AREA)
        self.img = image.copy()
        self.img2 = image.copy()
        image = image / np.max(image)
        image = image.swapaxes(0, 2)
        image = image.swapaxes(1, 2)
        image = np.array([image])
        image = torch.from_numpy(image)
        image = image.to(self.device)
        with torch.no_grad():
            char_mask, field_mask = self.model(image)
        char_mask = char_mask.cpu().detach().numpy()
        field_mask = field_mask.cpu().detach().numpy()
        fields = self.__get_fields(field_mask, threshold=self.field_threshold)
        preds = self.__get_texts(fields, char_mask)
        return preds
    
    
    
    def train(self, train_config_path: str) -> None:
        with open(train_config_path, encoding='UTF-8') as f:
            train_config = json.load(f)
        
        train_params = train_config['TRAIN']
        train_params['model'] = self.model

        train_data_generator = DataGenerator(**train_config['TRAIN_DATA_GENERATOR'])
        train_dataset_len = len(train_data_generator.img_names) // train_data_generator.batch_size

        if not os.path.exists(train_params['model_save_path']):
            os.mkdir(train_params['model_save_path'])

        if train_params['chars_weight']:
            chars_weight = np.load(train_params['chars_weight'])
            chars_weight = torch.tensor(chars_weight).to(self.device)
        else:
            chars_weight = None

        if train_params['fields_weight']:
            fields_weight = np.load(train_params['fields_weight'])
            fields_weight = torch.tensor(fields_weight).to(self.device)
        else:
            fields_weight = None
        
        loss_fn_1 = torch.nn.CrossEntropyLoss(weight=chars_weight)
        loss_fn_2 = torch.nn.CrossEntropyLoss(weight=fields_weight)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=train_params['lr'], momentum=train_params['momentum'])

        if 'less_than_loss' in train_params['save_condition']:
            best_loss = float(train_params['save_condition'][train_params['save_condition'].find(' '):])
        else:
            best_loss = 1000
                
        self.model.train()
        self.model.mode = 'train'

        for epoch in range(train_params['epochs']):
            train_loss1 = []
            train_loss2 = []            

            for i in tqdm(range(train_dataset_len)):
                optimizer.zero_grad()
                images, masks1, masks2 = train_data_generator.__getitem__(i)
                images = torch.tensor(images).to(self.device)
                masks1 = np.round(masks1)
                masks1 = np.array([np.argmax(a, axis=0) for a in masks1])
                masks1 = torch.tensor(masks1).to(self.device)
                masks2 = np.round(masks2)
                masks2 = np.array([np.argmax(a, axis=0) for a in masks2])
                masks2 = torch.tensor(masks2).to(self.device)
                outputs1, outputs2 = self.model(images)
                loss1 = loss_fn_1(outputs1.double(), masks1.long())
                loss2 = loss_fn_2(outputs2.double(), masks2.long())
                train_loss1.append(loss1.item())
                train_loss2.append(loss2.item())
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

            print(f'epoch: {epoch + 1}, char_loss: {np.mean(train_loss1)}, field_loss: {np.mean(train_loss2)}')

            if np.mean(loss) < best_loss:
                torch.save(self.model.state_dict(), os.path.join(train_params['model_save_path'],
                        f'epoch_{epoch + 1}_charloss_{round(np.mean(train_loss1), 3)}_fieldloss_{round(np.mean(train_loss2), 3)}.pt'))
                if train_params['save_condition'] == 'each_best_loss':
                    best_loss = np.mean(loss)
                elif train_params['save_condition'] == 'every_epoch':
                    best_loss = 1000
