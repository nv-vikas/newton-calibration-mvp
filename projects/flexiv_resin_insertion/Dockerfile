# Existing verified Isaac Lab 3.0 beta2 + Newton runtime. Keep MVP1 untouched.
FROM flexiv-peg-scene:agent-assist
WORKDIR /workspace/flexiv_peg_scene
COPY . /work/flexiv_resin_insertion
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/workspace/isaaclab/isaaclab.sh", "-p", "/work/flexiv_resin_insertion/run.py"]
