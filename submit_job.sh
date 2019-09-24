#!/usr/bin/env bash
gradient jobs create \
--name "$0" \
--container ${CONTAINER_NAME} \
--registryUsername ${DOCKERHUB_USERNAME} \
--registryPassword ${DOCKERHUB_PASSWORD} \
--machineType "G12" \
--command "./resolve.sh"