"""Gradio demo for SegCLIP open-vocabulary semantic segmentation.

Launch:
  python app_gradio.py --model-path checkpoints/segclip.bin --device cuda:0

This wraps the existing inference utilities found in main_seg_vis.py into an
interactive web UI supporting:
  * Image upload (any resolution)
  * Dataset class set selection (voc / context / coco)
  * Visualization mode selection (input, pred, input_pred, input_pred_label)

It lazily initializes the model on first request (or at startup) and caches
it for subsequent inferences.
"""
import argparse
import os
import os.path as osp
import time
from functools import lru_cache
from typing import List, Tuple

import gradio as gr
import mmcv
import torch

from omegaconf import read_write

# Local imports from project
from seg_segmentation.config import get_config
from seg_segmentation.evaluation import build_seg_inference, build_seg_demo_pipeline
from seg_segmentation.datasets import COCOObjectDataset, PascalContextDataset, PascalVOCDataset
from modules.modeling import SegCLIP
from modules.tokenization_clip import SimpleTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import collate, scatter
from mmcv.image import tensor2imgs
import numpy as np
import matplotlib.pyplot as plt


VIS_CHOICES = ["input", "pred", "input_pred", "input_pred_label"]
DATASET_CHOICES = ["voc", "context", "coco"]

class Tokenize:
    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate
    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True
        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(t) + [eot_token] for t in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[: self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError("Input too long")
            result[i, : len(tokens)] = torch.tensor(tokens)
        if expanded_dim:
            return result[0]
        return result

# Minimal OmegaConf default path
def default_cfg_path():
    return osp.join(osp.dirname(__file__), 'seg_segmentation', 'default.yml')

def _dataset_cfg(dataset_name: str) -> str:
    base = osp.join(osp.dirname(__file__), 'seg_segmentation', 'configs', '_base_', 'datasets')
    mapping = {
        'voc': 'pascal_voc12.py',
        'coco': 'coco.py',
        'context': 'pascal_context.py'
    }
    return osp.join(base, mapping[dataset_name])

DATASET_CLASS_MAP = {
    'voc': PascalVOCDataset,
    'coco': COCOObjectDataset,
    'context': PascalContextDataset
}

@lru_cache(maxsize=1)
def load_model(model_path: str, device: str, dataset_name: str, vis_mode: str, max_words: int = 77):
    # Capture outer scope variables by assigning as defaults inside the class body
    _cfg_path = default_cfg_path()
    _vis = [vis_mode]
    _device = device
    _dataset = dataset_name
    _max_words = max_words
    _model_path = model_path

    class Args:  # mimic argparse Namespace minimally
        cfg = _cfg_path
        opts = None
        resume = None
        vis = _vis
        device = _device
        dataset = _dataset
        input = ''
        output_dir = 'output'
        pretrained_clip_name = 'ViT-B/16'
        max_words = _max_words
        max_frames = 1
        init_model = _model_path
        cache_dir = ''
        seed = 42
        first_stage_layer = 10
        local_rank = 0
        output = 'output'
    args = Args()
    cfg = get_config(args)
    with read_write(cfg):
        cfg.evaluate.eval_only = True
        cfg.evaluate.seg.cfg = _dataset_cfg(dataset_name)
        # slide for arbitrary input
        cfg.evaluate.seg.opts = ['test_cfg.mode=slide']
    if model_path and osp.isfile(model_path):
        try:
            model_state_dict = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(f"Failed to load checkpoint {model_path}: {e}. Proceeding without weights.")
            model_state_dict = None
    else:
        print(f"Checkpoint {model_path} not found. Initializing model without pre-trained weights.")
        model_state_dict = None
    cache_dir = args.cache_dir if args.cache_dir else osp.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = SegCLIP.from_pretrained(cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    model.to(device)
    model = revert_sync_batchnorm(model)
    model.eval()
    text_transform = Tokenize(SimpleTokenizer(), max_seq_len=args.max_words)
    dataset_class = DATASET_CLASS_MAP[dataset_name]
    seg_model = build_seg_inference(model, dataset_class, text_transform, cfg.evaluate.seg)
    test_pipeline = build_seg_demo_pipeline(img_size=224)
    return seg_model, test_pipeline, device


def _run_inference(image: np.ndarray, seg_model, test_pipeline, device: str, vis_modes: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    data = dict(img=image)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(seg_model.parameters()).is_cuda:
        data = scatter(data, [torch.device(device)])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    with torch.no_grad():
        result = seg_model(return_loss=False, rescale=True, **data)
    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    outputs = []
    labels = []
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))
        for vis_mode in vis_modes:
            if vis_mode == 'input':
                outputs.append(img_show[:, :, ::-1])  # convert BGR to RGB
                labels.append('input')
            elif vis_mode == 'pred':
                palette = np.array(seg_model.PALETTE).astype(np.uint8)
                seg = result[0].astype(np.uint8)
                color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
                for label_idx, color in enumerate(palette):
                    color_seg[seg == label_idx, :] = color
                outputs.append(color_seg)
                labels.append('pred (palette)')
            elif vis_mode == 'input_pred':
                blend = seg_model.blend_result(img=img_show, result=result, out_file=None, opacity=0.7, with_bg=seg_model.with_bg)
                outputs.append(blend[:, :, ::-1])
                labels.append('input + pred')
            elif vis_mode == 'input_pred_label':
                # Inline implementation replicating show_result label drawing to avoid disk IO
                seg = result[0]
                labels_unique = np.unique(seg)
                coord_map = {}
                h_tmp, w_tmp = seg.shape
                coords = np.stack(np.meshgrid(np.arange(h_tmp), np.arange(w_tmp), indexing='ij'), axis=-1)
                for lab in labels_unique:
                    coord_map[lab] = coords[seg == lab].mean(axis=0)
                blended = seg_model.blend_result(img=img_show, result=result, out_file=None, opacity=0.6, with_bg=seg_model.with_bg)
                blended = mmcv.bgr2rgb(blended)
                width, height = img_show.shape[1], img_show.shape[0]
                fig = plt.figure(frameon=False)
                canvas = fig.canvas
                dpi = fig.get_dpi()
                fig.set_size_inches((width) / dpi, (height) / dpi)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                ax = plt.gca(); ax.axis('off')
                for lab in labels_unique:
                    if seg_model.with_bg and lab == 0:
                        continue
                    center = coord_map[lab].astype(np.int32)
                    ax.text(center[1], center[0], seg_model.CLASSES[lab],
                            bbox={'facecolor':'black','alpha':0.5,'pad':0.7,'edgecolor':'none'},
                            color='orangered', fontsize=14, va='top', ha='left')
                plt.imshow(blended)
                stream, _ = canvas.print_to_buffer()
                buffer = np.frombuffer(stream, dtype='uint8')
                img_rgba = buffer.reshape(height, width, 4)
                rgb, _alpha = np.split(img_rgba, [3], axis=2)
                final_img = mmcv.rgb2bgr(rgb)
                outputs.append(final_img[:, :, ::-1])  # back to RGB for gradio
                labels.append('input + pred + labels')
                plt.close(fig)
    return outputs, labels


def inference_api(image: np.ndarray, dataset: str, vis_modes: List[str], model_path: str, device: str):
    if image is None:
        return []
    seg_model, test_pipeline, device = load_model(model_path, device, dataset, vis_modes[0] if vis_modes else 'input')
    outputs, labels = _run_inference(image, seg_model, test_pipeline, device, vis_modes or ['input_pred_label'])
    return [(o, l) for o, l in zip(outputs, labels)]


def build_interface():
    with gr.Blocks(title="SegCLIP Semantic Segmentation") as demo:
        gr.Markdown("# SegCLIP Open-Vocabulary Semantic Segmentation\nUpload an image and choose the dataset class set.")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type='numpy', label='Input Image')
                dataset = gr.Radio(choices=DATASET_CHOICES, value='voc', label='Dataset classes')
                vis = gr.CheckboxGroup(choices=VIS_CHOICES, value=['input_pred_label'], label='Visualization modes')
                model_path = gr.Textbox(value='checkpoints/segclip.bin', label='Model checkpoint path')
                device = gr.Textbox(value='cuda:0' if torch.cuda.is_available() else 'cpu', label='Device')
                run_btn = gr.Button('Run Segmentation')
            with gr.Column(scale=2):
                gallery = gr.Gallery(label='Outputs', show_label=True, columns=2, height=600)
        run_btn.click(fn=inference_api, inputs=[image_input, dataset, vis, model_path, device], outputs=gallery)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='checkpoints/segclip.bin')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--server-name', type=str, default='0.0.0.0')
    parser.add_argument('--server-port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == '__main__':
    main()
