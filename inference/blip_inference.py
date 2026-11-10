import os

import time
import argparse
from tqdm import tqdm
import json
from PIL import Image


def main(args):

    print(f"curr used gpu: {args.device}")
    device_id = args.device.split(':')[-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration

    model_dir = args.model_dir

    model_names = ["blip2-opt-2.7b", "blip2-flan-t5-xl", "blip2-opt-6.7b", "blip2-flan-t5-xxl", ]

    image_folders = args.image_folders
    if not isinstance(image_folders, list) or not image_folders:
        raise ValueError("--image_folders must be a nonempty JSON list")
    for folder in image_folders:
        if not any(name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')) for name in os.listdir(folder)):
            raise ValueError(f"No supported images in {folder}")

    khits = args.khits

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

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Log directory '{log_dir}' created successfully")

    for model_name in model_names:
        model = None
        torch.cuda.empty_cache()

        model_name = os.path.join(model_dir, model_name)
        processor = Blip2Processor.from_pretrained(model_name)
        if "blip2-flan-t5-xxl" in model_name:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, load_in_8bit=True)
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')

        for image_folder in image_folders:
                target_responses = []
                for part in image_folder.split('-'):
                    if "response" in part:
                        sub_parts = part.split('_')[1].split(',')
                        for sub_part in sub_parts:
                            target_responses.append(sub_part.strip())

                out = []
                success = [0 for _ in range(khits)]
                for idx, image_file in enumerate(tqdm(sorted(name for name in os.listdir(image_folder) if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))))):
                    question = question_pool[idx % len(question_pool)]
                    question = "Question: " + question + " Answer:"

                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

                    inputs = processor(image, question, return_tensors="pt").to('cuda', torch.float16)
                    for key in inputs:
                        repeat_sizes = [khits] + [1] * (inputs[key].dim() - 1)
                        inputs[key] = inputs[key].repeat(*repeat_sizes)

                    outputs = model.generate(**inputs,
                                                do_sample=True,
                                                max_length=200,
                                                min_length=1,
                                                top_p=0.9,
                                                temperature=1,)

                    for i in range(len(outputs)):
                        answer = processor.decode(outputs[i], skip_special_tokens=True).strip()
                        out.append({'image file': image_file, 'prompt': question, 'continuation': answer})
                        if any(response.lower() in answer.lower() for response in target_responses):
                            for j in range(i, len(success)):
                                success[j] += 1
                            break

                total = len(sorted(name for name in os.listdir(image_folder) if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))))
                asr = "ASR: "
                for i, count in enumerate(success):
                    asr = asr + f" {i+1}hit({(count/total):.4f}) "
                print(asr)

                log_file = image_folder.split('/')[-1]
                log_file += f'-{khits}hits'
                log_file += f'-{os.path.basename(model_name)}'
                question_file_name = os.path.splitext(os.path.basename(questions_file))[0]
                log_file += f'-prompts_{question_file_name}.log'
                with open(f"{log_dir}/{log_file}", 'w') as f:
                    for li in out:
                        f.write(json.dumps(li))
                        f.write("\n")
                    f.write(asr)

    model_names = ["instructblip-flan-t5-xl", "instructblip-vicuna-7b", "instructblip-vicuna-13b"]

    for model_name in model_names:
        model = None
        torch.cuda.empty_cache()

        model_name = os.path.join(model_dir, model_name)
        if "instructblip-vicuna-13b" in model_name or "instructblip-flan-t5-xxl" in model_name:
            model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, load_in_8bit=True)
        else:
            model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')
        processor = InstructBlipProcessor.from_pretrained(model_name)

        for image_folder in image_folders:
                target_responses = []
                for part in image_folder.split('-'):
                    if "response" in part:
                        sub_parts = part.split('_')[1].split(',')
                        for sub_part in sub_parts:
                            target_responses.append(sub_part.strip())

                out = []
                success = [0 for _ in range(khits)]
                for idx, image_file in enumerate(tqdm(sorted(name for name in os.listdir(image_folder) if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))))):
                    question = question_pool[idx % len(question_pool)]
                    question = "Question: " + question + " Answer:"

                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

                    inputs = processor(image, question, return_tensors="pt").to('cuda', torch.float16)
                    for key in inputs:
                        repeat_sizes = [khits] + [1] * (inputs[key].dim() - 1)
                        inputs[key] = inputs[key].repeat(*repeat_sizes)

                    outputs = model.generate(
                            **inputs,
                            do_sample=True,
                            max_length=200,
                            min_length=1,
                            top_p=0.9,
                            repetition_penalty=1.5,
                            length_penalty=1.0,
                            temperature=1,
                    )

                    for i in range(len(outputs)):
                        answer = processor.decode(outputs[i], skip_special_tokens=True).strip()
                        out.append({'image file': image_file, 'prompt': question, 'continuation': answer})
                        if any(response.lower() in answer.lower() for response in target_responses):
                            for j in range(i, len(success)):
                                success[j] += 1
                            break

                total = len(sorted(name for name in os.listdir(image_folder) if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))))
                asr = "ASR: "
                for i, count in enumerate(success):
                    asr = asr + f" {i+1}hit({(count/total):.4f}) "
                print(asr)

                log_file = image_folder.split('/')[-1]
                log_file += f'-{khits}hits'
                log_file += f'-{os.path.basename(model_name)}'
                question_file_name = os.path.splitext(os.path.basename(questions_file))[0]
                log_file += f'-prompts_{question_file_name}.log'
                with open(f"{log_dir}/{log_file}", 'w') as f:
                    for li in out:
                        f.write(json.dumps(li))
                        f.write("\n")
                    f.write(asr)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folders', type=json.loads, required=True, help='JSON list of image directories (absolute paths for MiniGPT).')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--khits', type=int, default=10)
    parser.add_argument('--questions-file', type=str,
                        required=True,
                        help="Prompt file containing one nonempty prompt per line.")
    parser.add_argument('--log-dir', default='inference_logs')
    parser.add_argument('--model-dir', default='models', help='Directory containing the checkpoint subdirectories listed in this script.')
    args = parser.parse_args()
    main(args)
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) / 3600}h")
