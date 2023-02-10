sudo docker run -p 8501:8501 --mount type=bind,source="$(pwd)",target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving
