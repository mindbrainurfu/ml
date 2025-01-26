from pathlib import Path
import os
import locale
# import sys
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def process_and_run_commands(text_answer, work_path = Path('./save_fish') , sample_generation_number=1):
    reference_audio_path = work_path / 'audio' / 'reference_voice' / 'RU_Male_SpoungeBob'
    file_path = Path("save_fish/text/RU_Male_SpoungeBob.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        prompt_text = file.read()
    # prompt_text = "If you stay in bed. If you stay on the couch. If you stay in your comfort zone. If you only do what is easy your life will be hard. But if you do what is hard if you get up if you grind, if you are relentless, if you work as hard as possible, when other people are slacking off, your life will be easy."
    # text_answer = "Здравствуйте, чем могу помочь?"
    audio_chunk_length = 400
    device = 'cuda'
    # output_dir = work_path / "generated_audio"

    cmd_2 = f'python -m tools.llama.generate --text "{text_answer}" --prompt-text "{prompt_text}" --prompt-tokens \"{reference_audio_path}.npy" --checkpoint-path "fish_speech/checkpoints/fish-speech-1.5" \
    --num-samples {sample_generation_number} --chunk-length {audio_chunk_length} --device {device}'
    print(f"Executing: {cmd_2}")
    os.system(cmd_2)

    for i in range(0, sample_generation_number):
        cmd_3 = f"python -m tools.vqgan.inference  -i \"codes_{i}.npy\" --checkpoint-path \"fish_speech/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth\" -d {device}"
        print(f"Executing: {cmd_3}")
        os.system(cmd_3)
        
        os.system(f"rm codes_{i}.npy")
        
    output_audio_path = Path("fake.wav")

    if output_audio_path.exists():
        return output_audio_path
    else:
        raise FileNotFoundError("Generated audio file not found.")
    # return output_dir_path     

# def generate_reference_voice(work_path, generation_name, device='cuda'):
#         audio_path = work_path / 'audio' / 'reference_voice' / generation_name
#         cmd_1 = f"python -m tools.vqgan.inference -i \"{audio_path}.mp3\" --output-path \"{audio_path}.wav\" --checkpoint-path \"fish_speech/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth\" -d {device}"
#         print(f"Executing: {cmd_1}")
#         os.system(cmd_1)



# if __name__ == "__main__":
#     work_path = Path('./save_fish')   
#     # generate_reference_voice(work_path = Path('./save_fish') , generation_name='RU_Male_SpoungeBob')
#     if True:
#         process_and_run_commands(work_path)
        

