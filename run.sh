#!/bin/bash

NAME=arl_project
IMAGE=arl_project:latest

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

if ! [ -f "${XAUTH}" ]; then
  touch "${XAUTH}"
  xauth nlist "${DISPLAY}" | sed -e 's/^..../ffff/' | xauth -f "${XAUTH}" nmerge -
  chmod 644 "${XAUTH}"
fi

if [ "$(docker ps -aq -f status=exited -f name="${NAME}")" ]; then
  # Start the existing container and attach to it
  docker start "${NAME}"
  exec docker exec -it "${NAME}" bash
elif [ "$(docker ps -aq -f status=running -f name="${NAME}")" ]; then
  # Attach to the running container
  exec docker exec -it "${NAME}" bash
else
  # Create a new container and attach to it
  exec docker run \
    -it \
    --env=DISPLAY="${DISPLAY}"  \
    --env=QT_X11_NO_MITSHM=1 \
    --env=XDG_RUNTIME_DIR=/tmp \
    --env=XAUTHORITY="${XAUTH}" \
    --volume="${XAUTH}":"${XAUTH}" \
    --volume="${XSOCK}":"${XSOCK}" \
    --network=host \
    --privileged \
    --gpus=all \
    --name="${NAME}" \
    "${IMAGE}" \
    bash
fi
