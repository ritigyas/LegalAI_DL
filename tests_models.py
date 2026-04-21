import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyBToUXh04zpOTN_Ksa7m31Bkdft73487Do")

for m in genai.list_models():
    print(m.name, "->", m.supported_generation_methods)