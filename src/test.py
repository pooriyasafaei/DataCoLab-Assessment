# from transformers import PegasusTokenizer, PegasusForConditionalGeneration
#
#
# def summarize_text(text):
#     model_name = "google/pegasus-xsum"
#     tokenizer = PegasusTokenizer.from_pretrained(model_name)
#     model = PegasusForConditionalGeneration.from_pretrained(model_name)
#
#     tokens = tokenizer.encode(text, truncation=True, max_length=1024, return_tensors="pt")
#     summary_ids = model.generate(tokens, max_length=100, min_length=60, length_penalty=2.0, num_beams=4,
#                                  early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#
#     return summary
#
#
# # Example text of about 3000 characters long_text = """ ===The Use of Forc he Promotion of Human Right roblems of
# Distributive Justic n Ethics of World Order==="""
#
# summary = summarize_text(long_text)
# print("Summary:", summary)
#######################################################################################################################
# import torch
# from diffusers import StableDiffusionPipeline
#
#
# def generate_image_with_stable_diffusion(prompt):
#     pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
#
#     if torch.cuda.is_available():
#         pipe = pipe.to("cuda")
#     image = pipe(prompt, num_inference_steps=50,  # Increased from a default lower number
#                  guidance_scale=8  # Adjust based on how much you want to adhere to the prompt
#                  ).images[0]
#     return image
#
#
# # Example usage
# prompt = """The Cruel Sea: A book published in 1951, categorized under fiction, novel and written by Nicholas
# Monsarrat. The story: The story of the Royal Navys sinking of several German Uboats during World War Two is told from
# the point of view of the officers on board."""
# image = generate_image_with_stable_diffusion(prompt)
# image.show()
# image.save('sample-10-50.png')
