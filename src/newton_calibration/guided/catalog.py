"""User-facing recipe catalog. Listing a future recipe never installs an executor."""

from pathlib import Path

from newton_calibration.reporting import load_recipe

ARM_RECIPE = Path(__file__).resolve().parents[1] / "recipes/arm_joint_response.v1.json"
ARM_ID = "arm_joint_response@1"


def list_recipes():
    return [
        {
            "id": ARM_ID,
            "name": "Arm joint tuning",
            "status": "available",
            "description": "Supported joint-position response. Start with per-joint stiffness and damping; request other supported parameters explicitly.",
        },
        {
            "id": "grasp",
            "name": "Grasp calibration",
            "status": "planned",
            "description": "Not implemented. Will require gripper, object and contact evidence.",
        },
        {
            "id": "insertion",
            "name": "Insertion calibration",
            "status": "planned",
            "description": "Not implemented. Will require peg/hole, contact and task evidence.",
        },
    ]


def get_recipe(recipe_id):
    if recipe_id != ARM_ID:
        raise ValueError(f"Recipe {recipe_id!r} is not available; choose {ARM_ID}")
    return load_recipe(ARM_RECIPE)


def question(key, prompt, why, *, blocks="fit", example=None):
    return {"key": key, "question": prompt, "why": why, "blocks": blocks, "example": example}
