# Build from repository root. Thin offline layer; frozen task source in /work is unchanged.
ARG BASE_IMAGE=flexiv-resin-insertion:20260916-v24
FROM ${BASE_IMAGE}
COPY src/newton_calibration /feature/src/newton_calibration
COPY projects/flexiv_resin_insertion/qualification /feature/projects/flexiv_resin_insertion/qualification
COPY projects/flexiv_resin_insertion/tools /feature/projects/flexiv_resin_insertion/tools
ENV PYTHONPATH=/feature/src:/feature
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/workspace/isaaclab/_isaac_sim/python.sh", "-m", "newton_calibration.cli", "qualify"]
