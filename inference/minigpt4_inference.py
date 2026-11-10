"""MiniGPT-4 inference.

Install the external MiniGPT-4 repository in its own environment first:

    git clone https://github.com/Vision-CAIR/MiniGPT-4.git /absolute/path/to/MiniGPT-4
    cd /absolute/path/to/MiniGPT-4
    conda env create -f environment.yml
    conda activate minigptv
    export MINIGPT4_ROOT=/absolute/path/to/MiniGPT-4

Follow the upstream README to download the Vicuna V0 or Llama 2 Chat base
weights and the matching MiniGPT-4 checkpoint:
    https://github.com/Vision-CAIR/MiniGPT-4#getting-started

Set llama_model in the corresponding minigpt4/configs/models/ YAML and ckpt
in the evaluation YAML. Select that evaluation file with --cfg-path, e.g.
eval_configs/minigpt4_llama2_eval.yaml or eval_configs/minigpt4_eval.yaml.
Do not install this project's Hugging Face requirements.txt into the MiniGPT
environment; use the upstream environment.yml dependencies.

This script changes its working directory to MINIGPT4_ROOT. Use absolute
paths for --image_folders, --questions-file, and --log-dir. Evaluation images
should be downloaded and extracted into this project's dataset/ directory.
"""

import os
import sys

import argparse
import json
import time

# MiniGPT uses repository-relative model/config resources.
repo = os.environ.get("MINIGPT4_ROOT")
if not repo:
    raise RuntimeError("Set MINIGPT4_ROOT to the installed MiniGPT-4 repository.")
repo = os.path.abspath(os.path.expanduser(repo))
sys.path.insert(0, repo)
os.chdir(repo)

import torch
from tqdm import tqdm

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


class CustomChat(Chat):

    def get_context_emb(self, conv, img_list):

        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')

        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens] # text to embeddings

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, khits=10):

        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        with torch.inference_mode():
            outputs = self.model.llama_model.generate(
                inputs_embeds=embs.repeat(khits, 1, 1),
                max_new_tokens=max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=True,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )

        outputs_text = []
        for output_token in outputs:
            output_text = self.model.llama_tokenizer.decode(
                output_token, skip_special_tokens=True
            )
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_text = output_text.split('</s>')[0]
            outputs_text.append(output_text)

        return outputs_text


def main(args):

    model_cfgs = [args.cfg_path]

    image_folders = args.image_folders
    if not isinstance(image_folders, list) or not image_folders:
        raise ValueError("--image_folders must be a nonempty JSON list")
    for folder in image_folders:
        if not any(name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')) for name in os.listdir(folder)):
            raise ValueError(f"No supported images in {folder}")

    os.makedirs(args.log_dir, exist_ok=True)

    questions_file = args.questions_file
    question_pool = []
    if not questions_file:
        raise ValueError("--questions-file must specify a prompt file")
    with open(questions_file, 'r') as file:
        for line in file:
            if line.strip():
                question_pool.append(line.strip())
    if not question_pool:
        raise ValueError("Question file contains no nonempty prompts")

    khits = args.khits

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Log directory '{log_dir}' created successfully")

    for config in model_cfgs:
        args.cfg_path = config

        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0, 'pretrain_llama2': CONV_VISION_LLama2}

        print('Initializing Chat')

        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        CONV_VISION = conv_dict[model_config.model_type]

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        chat = CustomChat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
        print('Initialization Finished')

        for image_folder in image_folders:
            out = []
            success = [0 for _ in range(khits)]

            target_responses = []
            for part in image_folder.split('-'):
                if "response" in part:
                    sub_parts = part.split('_')[1].split(',')
                    for sub_part in sub_parts:
                        target_responses.append(sub_part.strip())

            for idx, image_file in enumerate(tqdm(sorted(name for name in os.listdir(image_folder) if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))))):
                prompt = question_pool[idx % len(question_pool)]

                chat_state = CONV_VISION.copy()
                img_list= []

                chat.upload_img(os.path.join(image_folder, image_file), chat_state, img_list)
                chat.ask(prompt, chat_state)
                chat.encode_img(img_list)
                try:

                    outputs_text = chat.answer(conv=chat_state,
                                                img_list=img_list,
                                                num_beams=2,
                                                top_p=0.9,
                                                temperature=1.0,
                                                max_new_tokens=150,
                                                max_length=1000,
                                                khits=khits)
                    for i in range(len(outputs_text)):
                        answer = outputs_text[i]
                        out.append({'image file': image_file, 'prompt': prompt, 'continuation': answer})
                        if any(response.lower() in answer.lower() for response in target_responses):
                            for j in range(i, len(success)):
                                success[j] += 1
                            break
                except Exception as exc:
                    raise RuntimeError(f"Inference failed for {image_file}") from exc

            total = len(sorted(name for name in os.listdir(image_folder) if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))))
            asr = "ASR: "
            for i, count in enumerate(success):
                asr = asr + f" {i+1}hit({(count/total):.4f}) "
            print(asr)

            log_file = image_folder.split('/')[-1]
            log_file += f'-{khits}hits'
            log_file += f'-minigpt4v1-{model_config.model_type}'
            question_file_name = os.path.splitext(os.path.basename(questions_file))[0]
            log_file += f'-prompts_{question_file_name}.log'
            with open(os.path.join(log_dir, log_file), 'w') as f:
                for li in out:
                    f.write(json.dumps(li))
                    f.write("\n")
                f.write(asr)

        del model, chat
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--image_folders', type=json.loads, required=True, help='JSON list of image directories (absolute paths for MiniGPT).')
    parser.add_argument('--questions-file', type=str,
                        required=True,
                        help="Prompt file containing one nonempty prompt per line.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--cfg-path", default='eval_configs/minigpt4_llama2_eval.yaml', help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config")

    start_time = time.time()
    parser.add_argument('--log-dir', default='inference_logs')
    parser.add_argument('--khits', type=int, default=10)
    args = parser.parse_args()
    main(args)
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) / 3600}h")
