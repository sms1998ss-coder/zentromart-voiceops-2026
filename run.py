"""Batch CLI evaluation runner for ZentroMart VoiceOps 2026 Capstone.

Processes a folder of customer .wav audio files through:
Gnani STT -> FAISS RAG + Google Gemini (gemini-3.5-flash-lite) -> Gnani TTS.
"""
import os
import argparse
from dotenv import load_dotenv
from src.rag import build_index
from src.speech import transcribe_audio, synthesize_speech
from src.agent import ZentroMartAgent

# Load GEMINI_API_KEY and GNANI_API_KEY from .env file
load_dotenv()


def process_audio_folder(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("1/3 Building Vector Index from kb/ folder...")
    retriever = build_index("./kb")

    print("2/3 Initializing Gemini Agent (gemini-3.5-flash-lite)...")
    agent = ZentroMartAgent(retriever=retriever)

    if not os.path.exists(input_dir):
        print(f"\nWaiting for files... Create '{input_dir}' folder and add customer .wav files.")
        return

    print("3/3 Processing Audio files...\n")
    wav_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"No .wav files found in '{input_dir}'. Please add .wav recordings and re-run.")
        return

    for file in wav_files:
        in_path = os.path.join(input_dir, file)
        out_audio = os.path.join(output_dir, f"reply_{file}")
        out_txt = os.path.join(output_dir, f"log_{file}.txt")

        print(f"Processing: {file}")

        # Step 4: Ears (Gnani STT)
        user_text = transcribe_audio(in_path)

        # Steps 1-3: Brain (Resetting history per file for independent batch testing)
        agent.reset_history()
        reply_text, sources = agent.chat(user_text)

        # Step 4: Voice (Gnani TTS)
        synthesize_speech(reply_text, out_audio)

        # Save execution logs
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(
                f"Customer STT: {user_text}\n\n"
                f"Agent Reply: {reply_text}\n\n"
                f"Sources Used: {sources}\n"
            )

    print("\nBatch processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps Batch Audio Processor")
    parser.add_argument("--input", default="sample_audio", help="Folder containing customer .wav files")
    parser.add_argument("--output", default="sample_outputs", help="Folder to save spoken replies and logs")
    args = parser.parse_args()
    process_audio_folder(args.input, args.output)
