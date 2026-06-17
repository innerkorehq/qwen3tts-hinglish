import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_ID = "ai4bharat/indic-parler-tts"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using:", device)

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # MPS safest
)

model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

text = """
भारत दुनिया की सबसे तेजी से बढ़ती अर्थव्यवस्थाओं में से एक है।
"""

description = """
A professional Indian male news anchor speaking confidently.
The recording is studio quality with clear pronunciation.
"""

description_inputs = description_tokenizer(description, return_tensors="pt")

prompt_inputs = tokenizer(text, return_tensors="pt")

description_inputs = {k: v.to(device) for k, v in description_inputs.items()}

prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

with torch.no_grad():
    generation = model.generate(
        input_ids=description_inputs["input_ids"],
        attention_mask=description_inputs["attention_mask"],
        prompt_input_ids=prompt_inputs["input_ids"],
        prompt_attention_mask=prompt_inputs["attention_mask"],
        do_sample=True,
        temperature=1.0,
    )

audio = generation.cpu().numpy().squeeze()

sf.write("output.wav", audio, model.config.sampling_rate)

print("Saved output.wav")
