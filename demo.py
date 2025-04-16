import torch
import argparse
from transformers import BertTokenizerFast
from models.HAMMER import HAMMER
import yaml
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import torch.nn.functional as F
from models import box_ops
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from io import BytesIO
import base64
import uvicorn
from fastapi import Form
import re

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载配置和模型
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/Pretrain.yaml')
parser.add_argument('--checkpoint', default='')
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--output_dir', default='/mnt/lustre/share/rshao/data/FakeNews/Ours/results')
parser.add_argument('--text_encoder', default='./bert_localpath/')
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--distributed', default=False, type=bool)
parser.add_argument('--rank', default=-1, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str)
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
parser.add_argument('--log_num', '-l', type=str)
parser.add_argument('--model_save_epoch', type=int, default=5)
parser.add_argument('--token_momentum', default=False, action='store_true')
parser.add_argument('--test_epoch', default='best', type=str)

args = parser.parse_args()

tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)

config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

model = HAMMER(
    args=args,
    config=config,
    text_encoder=args.text_encoder,
    tokenizer=tokenizer,
    init_deit=True
).to(device)

checkpoint = torch.load('HAMMER_checkpoint_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

app = FastAPI(title="HAMMER多模态推理API")


def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def preprocess_image(image):
    original_width, original_height = image.size
    image = image.convert("RGB")

    width, height = image.size
    if width > height:
        new_height = int(256 * height / width)
        image = image.resize((256, new_height))
        pad_top = (256 - new_height) // 2
        pad_bottom = 256 - new_height - pad_top
        image = ImageOps.expand(image, (0, pad_top, 0, pad_bottom))
        scale_x = original_width / 256
        scale_y = original_height / new_height
        pad_left = 0
    else:
        new_width = int(256 * width / height)
        image = image.resize((new_width, 256))
        pad_left = (256 - new_width) // 2
        pad_right = 256 - new_width - pad_left
        image = ImageOps.expand(image, (pad_left, 0, pad_right, 0))
        scale_x = original_width / new_width
        scale_y = original_height / 256
        pad_top = 0

    image = np.array(image).astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = image / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device).to(torch.float32)

    padding_info = {
        'original_width': original_width,
        'original_height': original_height,
        'pad_left': pad_left,
        'pad_top': pad_top,
        'scale_x': scale_x,
        'scale_y': scale_y
    }
    return image_tensor, padding_info


def preprocess_text(text):
    text_input = tokenizer(
        text,
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False
    )

    if isinstance(text_input.input_ids[0], int):
        input_ids = text_input.input_ids[:-1]
        attention_mask = text_input.attention_mask[:-1]
    else:
        input_ids = [x[:-1] for x in text_input.input_ids]
        attention_mask = [x[:-1] for x in text_input.attention_mask]

    max_len = max(len(x) for x in [input_ids]) if isinstance(input_ids, list) else len(input_ids)
    input_ids = [x + [0] * (max_len - len(x)) for x in [input_ids]] if isinstance(input_ids, list) else [input_ids] + [
        0] * (max_len - 1)
    text_input.input_ids = torch.LongTensor(input_ids).to(device)

    max_len = max(len(x) for x in [attention_mask]) if isinstance(attention_mask, list) else len(attention_mask)
    attention_mask = [x + [0] * (max_len - len(x)) for x in [attention_mask]] if isinstance(attention_mask, list) else [
                                                                                                                           attention_mask] + [
                                                                                                                           0] * (
                                                                                                                                   max_len - 1)
    text_input.attention_mask = torch.LongTensor(attention_mask).to(device)

    return text_input


def highlight_text(text, token_positions):
    def get_token_char_mapping(text):
        encoding = tokenizer(text,
                             return_offsets_mapping=True,
                             max_length=128,
                             truncation=True,
                             add_special_tokens=True)
        return encoding['offset_mapping']

    token_char_map = get_token_char_mapping(text)
    highlight_ranges = []

    for token_idx in token_positions:
        if token_idx < len(token_char_map):
            start, end = token_char_map[token_idx]
            if start != 0 or end != 0:  # 跳过特殊token
                highlight_ranges.append((start, end))

    def merge_ranges(ranges):
        if not ranges:
            return []
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        merged = [sorted_ranges[0]]
        for current in sorted_ranges[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        return merged

    merged_ranges = merge_ranges(highlight_ranges)

    highlighted = []
    prev_end = 0
    for start, end in merged_ranges:
        highlighted.append(text[prev_end:start])
        highlighted.append(f'**{text[start:end]}**')  # Markdown格式加粗
        prev_end = end
    highlighted.append(text[prev_end:])
    return ''.join(highlighted)


def visualize_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    for box in boxes:
        if len(box) == 4:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
    return image

def generate_interpretation(multi_cls_pred):
    """生成基础分类解释"""
    labels = ['face_swap', 'face_attribute', 'text_swap', 'text_attribute']
    activated = [labels[i] for i, pred in enumerate(multi_cls_pred) if pred == 1]
    return f"检测到篡改类型：{', '.join(activated) if activated else '未检测到明确篡改'}"


def get_image_annotation(multi_cls_pred):
    """生成图片篡改注释"""
    image_labels = {
        'face_swap': '面部篡改(替换)',
        'face_attribute': '属性修改'
    }
    detected = []
    if multi_cls_pred[0]:
        detected.append(image_labels['face_swap'])
    if multi_cls_pred[1]:
        detected.append(image_labels['face_attribute'])

    if detected:
        return "检测到图片高亮处发生" + "、".join(detected)
    return None

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    text: str = Form(...),
):


    # 处理图片
    image_data = await image.read()
    pil_image = Image.open(BytesIO(image_data))
    image_tensor, padding_info = preprocess_image(pil_image)

    # 处理文本
    text_input = preprocess_text(text)

    # 模型推理
    with torch.no_grad():
        try:
            logits_real_fake, logits_multicls, output_coord, logits_tok = model(
                image_tensor,
                ['orig'],
                text_input,
                torch.zeros((1, 4)).to(device),
                [[]],
                is_train=False
            )
        except Exception as e:
            import traceback
            return JSONResponse(
                status_code=500,
                content={"message": f"模型推理错误:\n{traceback.format_exc()}"}
            )

    # 坐标处理强化
    output_coord = output_coord.cpu().numpy().reshape(1, -1)  # 确保形状为(1,4)
    boxes = box_ops.box_cxcywh_to_xyxy(torch.tensor(output_coord)).numpy()

    # 坐标转换
    adjusted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = (x1 * 256 - padding_info['pad_left']) * padding_info['scale_x']
        y1 = (y1 * 256 - padding_info['pad_top']) * padding_info['scale_y']
        x2 = (x2 * 256 - padding_info['pad_left']) * padding_info['scale_x']
        y2 = (y2 * 256 - padding_info['pad_top']) * padding_info['scale_y']

        # 边界检查
        x1 = max(0, min(padding_info['original_width'], int(round(x1))))
        y1 = max(0, min(padding_info['original_height'], int(round(y1))))
        x2 = max(0, min(padding_info['original_width'], int(round(x2))))
        y2 = max(0, min(padding_info['original_height'], int(round(y2))))
        adjusted_boxes.append([x1, y1, x2, y2])


    # 处理文本高亮
    logits_tok = logits_tok.squeeze().cpu().numpy()
    token_predictions = logits_tok.argmax(axis=-1).flatten()
    fake_token_pos = np.where(token_predictions == 1)[0].tolist()
    highlighted_text = highlight_text(text, fake_token_pos)

    # 生成标注图片
    annotated_image = visualize_boxes(pil_image.copy(), np.array(adjusted_boxes))
    image_base64 = image_to_base64(annotated_image)

    # 分类结果处理
    label_names = ['face_swap', 'face_attribute', 'text_swap', 'text_attribute']
    multi_cls_pred = (logits_multicls.squeeze().cpu().numpy() > 0).astype(int)

    # 构建解释内容
    base_explanation = generate_interpretation(multi_cls_pred)
    interpretation_lines = [base_explanation]

    # 图片篡改注释
    image_annotation = get_image_annotation(multi_cls_pred)
    if image_annotation:
        interpretation_lines.append(image_annotation)

    return JSONResponse(content={
        "highlighted_text": highlighted_text,
        "annotated_image": f"data:image/png;base64,{image_base64}",
        "interpretation": interpretation_lines,
    })




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)