"""Capability-labelled recipes. Collection support never installs a fitter."""

from pathlib import Path

from newton_calibration.collection.contact import load_collection_recipe
from newton_calibration.reporting import load_recipe

ARM_RECIPE = Path(__file__).resolve().parents[1] / "recipes/arm_joint_response.v1.json"
ARM_ID = "arm_joint_response@1"
CONTACT_RECIPES = {
    "grasp_contact@1": ARM_RECIPE.with_name("grasp_contact.v1.json"),
    "peg_insertion@1": ARM_RECIPE.with_name("peg_insertion.v1.json"),
}
ALIASES = {"grasp": "grasp_contact@1", "insertion": "peg_insertion@1"}


def list_recipes():
    return [
        {
            "id": ARM_ID,
            "name": "Arm joint tuning",
            "status": "available",
            "description": "Supported joint-position response. Start with per-joint stiffness and damping; request other supported parameters explicitly.",
        },
        {
            "id": "grasp_contact@1",
            "name": "Grasp calibration",
            "status": "collection_only",
            "description": "MVP2 setup questions, evidence contract, lab checklist and trial templates. No fitting or executable robot motions.",
        },
        {
            "id": "peg_insertion@1",
            "name": "Insertion calibration",
            "status": "collection_only",
            "description": "MVP3 setup questions, force/depth evidence contract, lab checklist and trial templates. No fitting or executable robot motions.",
        },
    ]


def get_recipe(recipe_id):
    recipe_id = ALIASES.get(recipe_id, recipe_id)
    if recipe_id in CONTACT_RECIPES:
        return load_collection_recipe(CONTACT_RECIPES[recipe_id])
    if recipe_id != ARM_ID:
        raise ValueError(f"Recipe {recipe_id!r} is not available; inspect guide recipes for supported capabilities")
    return load_recipe(ARM_RECIPE)


def question(key, prompt, why, *, blocks="fit", example=None):
    return {"key": key, "question": prompt, "why": why, "blocks": blocks, "example": example}
