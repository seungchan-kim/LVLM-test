docker run -it --rm \
	--gpus all \
	--network host \
	--ipc host \
	--privileged \
	--runtime=nvidia \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e ROS_DOMAIN_ID=1 \
	-v $(pwd):/app \
	-v ~/.cache/huggingface:/root/.cache/huggingface \
	lvlm:latest
