import os
import glob
import time
import argparse
import json
import logging
from typing import List
from tqdm import tqdm
from faster_whisper import WhisperModel

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Blazing Fast B200 Transcription via faster-whisper')
    parser.add_argument('--input_dir',  type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--language',   type=str, required=True, help='Language code (e.g., "pl", "cs")')
    # For 15-minute files, a device_index or workers allocation handles concurrency natively
    parser.add_argument('--model_id',   type=str, default="large-v3-turbo", help='Options: large-v3, large-v3-turbo')
    return parser.parse_args()

def get_audio_files(input_dir: str) -> List[str]:
    files = glob.glob(os.path.join(input_dir, "*"))
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    return [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in audio_extensions]

def main():
    setup_logging()
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info(f"Loading {args.model_id} onto Blackwell Tensor Cores...")
    # B200 Optimization: Native bfloat16 computation execution.
    # We set CPU threads to 4 to speed up file loading in parallel.
    model = WhisperModel(
        args.model_id, 
        device="cuda", 
        device_index=0, 
        compute_type="bfloat16", 
        cpu_threads=4
    )
    
    audio_files = get_audio_files(args.input_dir)
    logging.info(f"Found {len(audio_files)} audio files.")

    for audio_file in tqdm(audio_files, desc="Transcribing"):
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(args.output_dir, f"{filename}.json")
        
        if os.path.exists(output_file):
            logging.info(f"Skipping already processed file: {filename}")
            continue
            
        start_time = time.time()
        
        try:
            # beam_size=1 switches from slow beam search to fast greedy decoding (massive speed boost)
            segments, info = model.transcribe(
                audio_file, 
                language=args.language, 
                beam_size=1,
                word_timestamps=True
            )
            
            # Unpack segments generator into structured dictionary format
            chunks = []
            full_text = []
            for segment in segments:
                full_text.append(segment.text)
                chunks.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end
                })
                
            elapsed_time = time.time() - start_time
            
            output_data = {
                "audio_file": os.path.basename(audio_file),
                "full_text": "".join(full_text).strip(),
                "chunks": chunks,
                "processing_time": elapsed_time,
                "language": args.language,
                "model_id": args.model_id,
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Finished {filename} in {elapsed_time:.2f}s")
            
        except Exception as e:
            logging.error(f"Failed to process {audio_file}. Error: {e}")

if __name__ == "__main__":
    main()