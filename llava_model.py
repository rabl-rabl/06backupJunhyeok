from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

import torch
import re
    
class llava_next_video():
    ####https://huggingface.co/collections/llava-hf/llava-next-video-6666a9173a64c7052930f153
    
    def __init__(self):

        self.generate_kwargs = {"max_new_tokens": 100, "do_sample": True, "top_p": 0.9}
        self.device = 'cuda'
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            low_cpu_mem_usage=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-DPO-hf")
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf",
            quantization_config=self.quantization_config,
            device_map='auto'
        )

        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        

    def prompt_processing(self, conversation, video):
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        self.inputs = self.processor(text=prompt, videos=video, return_tensors="pt").to(self.device)
        return self.inputs


    def video_inference(self):              #비디오를 input받아서 뭔가 생성해내는 듯?
        out = self.model.generate(**self.inputs, **self.generate_kwargs)            
        out_input_string = str(self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True))[:-2]
        # Use regular expression to find the category name after "ASSISTANT:"\
        match = re.search(r"ASSISTANT:\s*(.*)", out_input_string)
        return match.group(1)