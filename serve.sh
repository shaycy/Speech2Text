sudo docker run -d -p 80:8501 --mount type=bind,source="$(pwd)",target=/models/speech2text -e MODEL_NAME=speech2text -t tensorflow/serving


curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://13.95.7.205/v1/models/speech2text:predict
