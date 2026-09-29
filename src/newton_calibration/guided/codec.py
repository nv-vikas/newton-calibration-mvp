"""Typed reload of the toolkit's own records; never executes serialized code."""

from newton_calibration.core.models import (
    AnalysisResult,
    CalibrationPlan,
    CandidateEvaluation,
    EnvironmentSpec,
    FitResult,
    ParameterSpec,
    ValidationResult,
)


def analysis(value):
    return AnalysisResult(
        **{
            **value,
            "environment": EnvironmentSpec(**value["environment"]),
            "identifiable_parameters": [ParameterSpec(**p) for p in value["identifiable_parameters"]],
        }
    )


def plan(value):
    return CalibrationPlan(
        **{
            **value,
            "environment": EnvironmentSpec(**value["environment"]),
            "parameters": [ParameterSpec(**p) for p in value["parameters"]],
        }
    )


def fit(value):
    return FitResult(
        **{
            **value,
            "plan": plan(value["plan"]),
            "baseline": CandidateEvaluation(**value["baseline"]),
            "best": CandidateEvaluation(**value["best"]),
        }
    )


def validation(value):
    return ValidationResult(**{**value, "fit": fit(value["fit"])})
