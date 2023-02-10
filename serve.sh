sudo docker run -d -p 8501:8501 --mount type=bind,source="$(pwd)",target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving


curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://13.95.7.205:8501/v1/models/my_model:predict
