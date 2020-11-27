#!/bin/bash

docker build --file Dockerfile --tag tacotron .
docker run -it --gpus all --ipc=host -p 8080:8080 -v /home/$USER/TTS/:/home/$USER tacotron:latest bash
