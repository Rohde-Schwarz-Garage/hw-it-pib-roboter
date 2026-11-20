# This Dockerfile expects paths relative to the project root.
FROM ros:jazzy-ros-base

# use bash by default
SHELL ["/bin/bash", "-c"]

# update and install packages and clean up
RUN apt-get update && \
    apt-get full-upgrade -y && \
    rosdep update --rosdistro $ROS_DISTRO && \
    apt-get install python3-pip pipewire pipewire-pulse wireplumber libportaudio2 alsa-utils espeak-ng pipewire-alsa unzip -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install datatypes ROS 2 package from https://github.com/pib-rocks/pib-backend/tree/main/ros_packages/datatypes
COPY pib-backend/ros_packages/datatypes /opt/pib-datatypes
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    cd /opt/pib-datatypes && \
    colcon build

COPY . /hw-it-pib-roboter

WORKDIR /hw-it-pib-roboter/

RUN pip install -e . --break-system-package

# install piper models:
# - https://huggingface.co/rhasspy/piper-voices/tree/main/de/de_DE/thorsten/high
# - https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ryan/high

RUN mkdir -p /opt/models/piper && \
    curl -fL -o /opt/models/piper/de_DE-thorsten-high.onnx 'https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx?download=true' && \
    curl -fL -o /opt/models/piper/de_DE-thorsten-high.onnx.json 'https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json?download=true' && \
    curl -fL -o /opt/models/piper/en_US-ryan-high.onnx 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx?download=true' && \
    curl -fL -o /opt/models/piper/en_US-ryan-high.onnx.json 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json?download=true'

# install vosk models
# - https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip
# - https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip

RUN mkdir -p /opt/models/vosk && \
    curl -fL -o /opt/models/vosk/de_small_0.15.zip 'https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip' && \
    unzip /opt/models/vosk/de_small_0.15.zip -d /opt/models/vosk/ && \
    mv /opt/models/vosk/vosk-model-small-de-0.15 /opt/models/vosk/de_small_0.15 && \
    rm /opt/models/vosk/de_small_0.15.zip && \
    curl -fL -o /opt/models/vosk/en-us_lgraph_0.22.zip 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip' && \
    unzip /opt/models/vosk/en-us_lgraph_0.22.zip -d /opt/models/vosk/ && \
    mv /opt/models/vosk/vosk-model-en-us-0.22-lgraph /opt/models/vosk/en-us_lgraph_0.22 && \
    rm /opt/models/vosk/en-us_lgraph_0.22.zip

WORKDIR /hw-it-pib-roboter/src/control


CMD [ "/hw-it-pib-roboter/src/control/docker-entrypoint.sh" ]