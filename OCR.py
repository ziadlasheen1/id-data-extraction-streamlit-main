from google import genai
import os
client = genai.Client(api_key="AIzaSyBx4efXkEnHsqn-PGsGtMRAJe3XvlK8Mi8")

map_id = {
    "name": "first name",
    "number": "id number",
    "Code":"code",
    "family name":"full name",
    "state":"state",
    "image":"image",
    "city":"city",
    "neighborhood":"neighborhood",
}

id_prompt = "\n".join([
    "the id contains 15 arabic numbers only no dashes or dots"
])
def extract_text(crop_folder):
    print("Extracting text from images...")
    extracted = {}
    for image in os.listdir(crop_folder):
        id_class = image.split(".")[0]
        if image == ".ipynb_checkpoints" or id_class == "image":
            continue
        id_class = map_id[id_class]
        prompt = "\n".join([
            "you are a professional arabic OCR model",
            "extract the data form this part of the egyption id"
            f"extract the {id_class}",
            "return the extracted information only",
            "don't generate any introductions or conclusions return just the data",
        ])
        if id_class == "id number":
            prompt += "\n" + id_prompt
        my_file = client.files.upload(file=os.path.join("croped",image))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[my_file, prompt]
        )
        extracted[id_class] = response.text
        print(extracted)
    return extracted
