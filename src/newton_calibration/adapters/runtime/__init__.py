from .analytic import AnalyticPDReplayRuntime

__all__ = ["AnalyticPDReplayRuntime"]


def create_runtime(environment):
    if environment.controller_profile:
        from newton_calibration.controllers import ControllerProfile, controller_report

        if not controller_report(ControllerProfile.from_dict(environment.controller_profile))["fit_ready"]:
            raise ValueError("Cannot create a fitting runtime for an unsupported or unconfirmed controller profile")
    if environment.adapter == "analytic":
        return AnalyticPDReplayRuntime(environment)
    if environment.adapter == "isaaclab_newton":
        from .newton import IsaacLabNewtonRuntime

        return IsaacLabNewtonRuntime(environment)
    raise KeyError(f"Unknown runtime adapter: {environment.adapter}")
