import os
import glob
import time
import argparse
import json
import logging
from typing import List
from tqdm import tqdm
import pandas as pd
import torch

# B200 Optimization 1: Enable Tensor Float 32 (TF32) execution on Blackwell Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='High-Throughput B200 Audio Transcription using Whisper')
    parser.add_argument('--input_dir',  type=str, required=True, help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for transcription JSON files')
    parser.add_argument('--language',   type=str, required=True, help='Language code (e.g., "pl", "cs")')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for B200 (Default: 128)')
    parser.add_argument('--model_id',   type=str, default="openai/whisper-large-v3-turbo", help='Whisper model ID')
    parser.add_argument('--return_timestamps', type=str, default='word', choices=['word', 'chunk'], help='Timestamp granularity')
    return parser.parse_args()

def setup_model(model_id: str, batch_size: int):
    # CRITICAL FIX: Removed hardcoded CUDA device overrides to respect command-line arguments properly.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    logging.info(f"Initializing Blackwell-optimized pipeline on device: {device}, dtype: {torch_dtype}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",  # Scaled Dot-Product Attention (Native FlashAttention on Blackwell)
    )
    model.to(device)
    model.eval()

    from transformers import AutoProcessor, pipeline
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=batch_size,  # B200 Optimization 2: Dynamically bind the scaled-up batch size
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def get_audio_files(input_dir: str) -> List[str]:
    # print(os.listdir(input_dir))
    files = glob.glob(os.path.join(input_dir, "*"))
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    return [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in audio_extensions]

# B200 Optimization 3: Asynchronous Dataset Generator. 
# Passing a generator directly to the Hugging Face pipeline allows it to stream data 
# using internal multi-threading, keeping the B200 GPU completely saturated.
def audio_generator(file_list: List[str]):
    for file_path in file_list:
        yield file_path

def process_audio_pipeline(pipe, audio_files: List[str], output_dir: str,
                           language: str, batch_size: int, model_id: str, return_timestamps: str):
    os.makedirs(output_dir, exist_ok=True)

    # Filter out already-processed files upfront to avoid pipeline contamination
    files_to_process = []
    for audio_file in audio_files:
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.json")
        if not os.path.exists(output_file):
            files_to_process.append(audio_file)
        else:
            logging.info(f"Skipping already processed file: {audio_file}")

    if not files_to_process:
        logging.info("All files already processed.")
        return

    logging.info(f"Beginning parallel streaming inference for {len(files_to_process)} files...")
    start_time = time.time()

    try:
        # B200 Optimization 4: Strict inference mode to clean up VRAM overhead
        with torch.inference_mode():
            pipeline_outputs = pipe(
                audio_generator(files_to_process),
                return_timestamps=return_timestamps,
                generate_kwargs={"language": language},
                batch_size=batch_size
            )

            # Iterate over the pipeline's streaming outputs (order is guaranteed to match files_to_process)
            for audio_file, result in zip(files_to_process, tqdm(pipeline_outputs, total=len(files_to_process), desc="B200 Transcribing")):
                elapsed_time = time.time() - start_time
                
                output_data = {
                    "audio_file":      os.path.basename(audio_file),
                    "full_text":       result["text"],
                    "chunks":          result["chunks"],
                    "processing_time": elapsed_time,
                    "language":        language,
                    "model_id":        model_id,
                }
                
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logging.error(f"Critical error during streaming execution: {e}")

def main():
    setup_logging()
    args = parse_arguments()
    
    pipe = setup_model(args.model_id, args.batch_size)
    audio_files = get_audio_files(args.input_dir)
    logging.info(f"Found {len(audio_files)} total audio files in destination directory.")
    
    if not audio_files:
        logging.error("No valid audio formats discovered.")
        return

    process_audio_pipeline(
        pipe,
        audio_files,
        args.output_dir,
        args.language,
        args.batch_size,
        args.model_id,
        args.return_timestamps
    )
    logging.info("Blackwell pipeline optimization task finished.")

if __name__ == "__main__":
    from transformers import AutoModelForSpeechSeq2Seq
    main()