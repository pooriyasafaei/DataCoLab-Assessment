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
# # Example text of about 3000 characters
# long_text = """ Joe Bonham, a young soldier serving in World War I, awakens in a hospital bed after being caught in the blast of an exploding artillery shell. He gradually realizes that he has lost his arms, legs, and all of his face (including his eyes, ears, teeth, and tongue), but that his mind functions perfectly, leaving him a prisoner in his own body. Joe attempts suicide by suffocation, but finds that he had been given a tracheotomy which he can neither remove nor control. At first Joe wishes to die, but later decides that he desires to be placed in a glass box and toured around the country in order to show others the true horrors of war. After he successfully communicates with his doctors by banging his head on his pillow in Morse code, however, he realizes that neither desire will be granted; it is implied that he will live the rest of his natural life in his condition. As Joe drifts between reality and fantasy, he remembers his old life with his family and girlfriend, and reflects upon the myths and realities of war. He also forms a bond, of sorts, with a young nurse who senses his plight.
# """
# summary = summarize_text(long_text)
# print("Summary:", summary)
#######################################################################################################################
from diffusers import StableDiffusionPipeline


def generate_image_with_stable_diffusion(prompt):
    # Load the model; requires Hugging Face authorization token
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
    pipe = pipe.to("cuda")  # Move the pipeline to GPU if available

    # Generate an image
    image = pipe(prompt).images[0]
    image.save("generated_image.png")  # Save the generated image

    return image


# Example usage
prompt = "A futuristic cityscape, illuminated by neon lights, reflecting a dystopian novel's theme."
image = generate_image_with_stable_diffusion(prompt)
image.show()

