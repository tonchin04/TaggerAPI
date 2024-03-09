from fastapi import FastAPI, UploadFile, Request
from contextlib import asynccontextmanager
from pydantic import model_serializer, BaseModel
import uvicorn
import cv2
import io
import numpy
from PIL import Image
import onnxruntime
import huggingface_hub
import json
import pandas
import msgpack

global models
models = []

def load_model(model_repo) -> onnxruntime.InferenceSession:
  path = huggingface_hub.hf_hub_download(repo_id=model_repo, filename="model.onnx", cache_dir="cache")
  model = onnxruntime.InferenceSession(path)
  return model


def load_labels() -> list[str]:
  tags = huggingface_hub.hf_hub_download(
      repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2", filename="selected_tags.csv", cache_dir="cache"
  )
  df = pandas.read_csv(tags)

  tag_names = df["name"].tolist()
  rating_indexes = list(numpy.where(df["category"] == 9)[0])
  general_indexes = list(numpy.where(df["category"] == 0)[0])
  character_indexes = list(numpy.where(df["category"] == 4)[0])
  return tag_names, rating_indexes, general_indexes, character_indexes


def square_resize(image, size):
  old_size = image.shape[:2]
  new_size = max(old_size)
  new_size = max(new_size, size)

  d_w = new_size - old_size[1]
  d_h = new_size - old_size[0]

  top, bottom = d_h // 2, d_h - (d_h // 2)
  left, right = d_w // 2, d_w - (d_w // 2)

  color = [255, 255, 255]
  new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)

  if new_image.shape[0] > size:
    new_image = cv2.resize(new_image, (size, size), interpolation = cv2.INTER_AREA)
  elif new_image.shape[0] < size:
    new_image = cv2.resize(new_image, (size, size), interpolation = cv2.INTER_CUBIC)

  return new_image


def image_convert(model, image):
  _, height, width, _ = model.get_inputs()[0].shape

  image = image.convert("RGBA")
  new_image = Image.new("RGBA", image.size, "WHITE")
  new_image.paste(image, mask=image)
  image = new_image.convert("RGB")
  image = numpy.asarray(image)

  image = image[:, :, ::-1]

  image = square_resize(image, height)
  image = image.astype(numpy.float32)
  image = numpy.expand_dims(image, 0)

  return image


def predict(
  image,
  model,
  is_detect_character,
  tag_threshold,
  character_threshold
  ):
  print("image received.")
  
  input_name = model.get_inputs()[0].name
  label_name = model.get_outputs()[0].name
  probs = model.run([label_name], {input_name: image})[0]

  labels = list(zip(tag_names, probs[0].astype(float)))

  ratings_names = [labels[i] for i in rating_indexes]
  rating = dict(ratings_names)

  general_names = [labels[i] for i in general_indexes]
  general_res = [x for x in general_names if x[1] > tag_threshold]
  general_res = dict(general_res)

  character_names = [labels[i] for i in character_indexes]
  character_res = [x for x in character_names if x[1] > character_threshold]
  character_res = dict(character_res)

  detected_tags = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))

  tags = (
    ", ".join(list(detected_tags.keys()))
    .replace("_", " ")
    #.replace("(", "\(")
    #.replace(")", "\)")
  )

  results = {
    #"tags": tags,
    "rating": rating,
    "general_res": general_res
  }

  if is_detect_character:
    results["character"] = character_res

  return results


def find_model(model_list, model_name):
  return model_list.index(model_name) if model_name in model_list else -1


@asynccontextmanager
async def lifespan(app: FastAPI):
  print("Tagger API is starting...")

  count = 0
  with open("./repos.json", "r") as f:
    repos = json.load(f)
  for repo in repos['repos']:
    if repo['isload'] == True:
      model = load_model(repo['repo'])
      models.append({"name": repo['name'], "model": model})
      print(repo['name'] + " is loaded.")
      count += 1
  if count > 1:
    print(count + " models are loaded.")
  global tag_names, rating_indexes, general_indexes, character_indexes
  tag_names, rating_indexes, general_indexes, character_indexes = load_labels()
  print("tags loaded.")

  yield
  print("Tagger API is shutting down...")
  #TODO: unload models

app = FastAPI(lifespan = lifespan)


@app.post("/api/tagger")
async def tagger(request: Request):
  raw_binary = await request.body()
  data = msgpack.unpackb(raw_binary, raw = False)
  try:
    with Image.open(io.BytesIO(data['image'])) as image:
      if data['model'] != None:
        model_list = [d.get('name') for d in models]
        model_index = find_model(model_list, data['model'])
        if model_index == -1:
          model = models[0]['model']
        else:
          model = models[model_index]['model']
      else:
        model = models[0]['model']

      if data['tag_threshold'] != None:
        tag_threshold = data['tag_threshold']
      else:
        tag_threshold = 0.5

      if data['is_detect_character'] == True:
        is_detect_character = True
        if data['character_threshold'] != None:
          character_threshold = data['character_threshold']
        else:
          character_threshold = 0.8

      else:
        is_detect_character = False
      image = image_convert(model, image)
      results = predict(image, model, is_detect_character, tag_threshold, character_threshold)

    return results

  except Exception as e:
    return {"error": str(e)}


if __name__ == "__main__":
  uvicorn.run(app, host = "0.0.0.0", port = 8080)
