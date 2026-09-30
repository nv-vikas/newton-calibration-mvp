# MVP2 grasp and MVP3 insertion: lab collection guide

**Purpose:** bring back replayable evidence, not just a successful demonstration.
The recipes are robot-agnostic. The notes below use the user's Flexiv + Grav +
peg/hole setup. A wrist sensor is reported present and an iPhone can be mounted;
sensor identity, exported channels, synchronization and squeeze-force reference
remain unconfirmed. Do not assume a different controller than the real run uses.

**Implemented now:** recipe selection, setup intake, source/version tracking,
parameter-to-evidence requirements, pilot run sheets, per-trial templates and a
printable HTML checklist. **Not implemented:** contact-data ingestion/audit,
contact fitting, numeric contact trajectories, Newton contact screening/preview,
robot execution, learned policy training or real transfer validation.

## What to tell the technician

> We want to record what the robot was actually commanded to do and how the peg
> responded. First do one reviewed grasp/hold/replace trial, then one reviewed
> centered insertion. Record actual arm and gripper commands, joint/TCP feedback,
> wrench and synchronized video. Inspect those pilot files before collecting the
> rest. You select and approve all speeds, forces, offsets and stop procedures.

For MVP2 the **robot gripper holds the peg** during lift/hold/transport. Record
the closing, grasp and release too. MVP3 starts with the peg already grasped,
with its offset in the fingers recorded. Nobody should hold the peg or hole
block by hand while the robot moves; use an operator-reviewed fixture/support.

## Bring and confirm before motion

| Item | What we need | Why |
|---|---|---|
| Robot engineer | Actual deployment mode/control chain and approved collection program, limits, stop/recovery and clear workspace | A Cartesian policy cannot be qualified by substituting joint-PD collection. |
| Gripper/tool setup | Gripper and finger identity, opening/squeeze convention, TCP, mounted mass/payload, configuration snapshot | Holding a peg changes the setup from the previous empty-gripper MVP1 data. |
| Actual peg + selected hole | Unique IDs, measured mass/dimensions, material/finish, chamfer/depth, fixture and frame transforms | CAD nominal values and the printed hole label are not as-built metrology. Use gauges with suitable uncertainty for the actual clearance. |
| Robot logger | Final dispatched commands, q/dq, TCP/flange state, controller versions/configuration, native timestamps and host times | Replaying the requested command may differ from what the driver published after filtering/limiting. |
| Wrist sensor | Identity, actual signal export, frames/origins, calibration/tare, compensation/filtering and freshness | Needed for quantitative insertion-load comparison. Presence of a data field is not proof a sensor is installed. |
| iPhone on fixed mount | Peg/fingers and hole mouth visible, original video plus timing/sync reference; calibrated scale/view for measured distances | Useful for slip, contact, depth and outcomes. A video alone cannot measure force or guarantee submillimeter pose. |
| Squeeze-force reference if available | Calibrated compression load cell/force gauge, instrumented jaws, or a qualified force map for the actual fingers | Required to separate jaw force from finger–peg friction. Wrist wrench cannot measure opposing jaw squeeze. |

One well-positioned fixed oblique video is a useful starting point. A second
side view can help when fingers obscure insertion depth. Have the technician
verify visibility with the actual grasp before collecting the suite. Preserve
original files and presentation timestamps; do not rely on file creation time
for synchronization. Record a common visible/logged event and clock uncertainty.
Keep robot and force streams at their native exported cadence, not camera cadence.
Never upsample a slow or stale channel and present it as new measurements.

## MVP2: what to collect

| Family | Starter count | Real evidence | What it can support |
|---|---:|---|---|
| Grasp → short lift → hold → transport → replace/release; two reviewed squeeze settings × two reviewed acceleration profiles | 8 development trials | Final arm/gripper commands, q/dq/TCP, actual jaw opening, video of peg relative to fingers; qualified wrench if available | Grasp timing, holding/slip behavior, losses of grip and transport response. |
| New grasp/transport conditions reserved before fitting | 4 held-out trials | Same signals; different predeclared conditions/profiles, entire separate episodes | Independent behavior validation; not a large-sample transfer claim. |
| Guarded/support-contained slip onset at two known normal loads | 4 **conditional** trials | Independent jaw normal load, tangential load and relative slip | Friction only if loads/geometry are identifiable. Do not provoke uncontrolled drops. |

The 12 core trials are a **pilot design**, not evidence that every parameter is
identifiable. Record failures and aborts as well as successes. Before fitting,
inspect repeatability and sensitivity; additional sliding or force/deflection
experiments may be required for kinetic friction, compliance or damping.

### If you cannot get a load cell

| Alternative | What it provides | Limit |
|---|---|---|
| Calibrated compression force gauge in an engineer-designed fixture | Independent squeeze-force reference | Must suit the jaw gap, loads and contact geometry; record per-jaw versus total convention. |
| Calibrated instrumented jaws / tactile force sensor | Normal load at the contact | Uncalibrated pressure values or arbitrary units are not force. Added hardware may change the grasp. |
| Validated gripper force map | Force estimate from command/current/gap under specified conditions | Must apply to this gripper, finger leverage, opening and loading regime; retain uncertainty. A nominal catalog maximum is not a map. |
| Logs + video only | Useful grasp, slip, lift and release evidence | Fit/validate effective behavior later if supported; **do not report a unique friction coefficient**. |

A hanging known mass or a pull gauge may provide **tangential load**, not jaw
normal force. With guarded operator-approved fixtures it can help bound holding
capacity, but does not separate weak squeezing from low friction by itself.

## MVP3: what to collect

Start with **one selected hole**, measured and identified. Do not introduce new
holes, material batches or controllers midway through a setup revision.

| Family | Starter count | Real evidence |
|---|---:|---|
| Centered insertion with pre-grasped peg | 2 development trials | Actual commands, q/dq/TCP, jaw opening, qualified wrench, relative peg/hole depth and video. |
| Positive/negative X offset, Y offset, tilt about X, tilt about Y | 16 development trials | Same; numerical offsets/angles and board axes are measured and operator-approved, never generic defaults. |
| Centered insertion at a second approved timing profile | 2 development trials | Same; preserve timing and controller settings. |
| Three new predeclared alignment/timing conditions | 6 held-out trials | Same; do not expose these episodes to fitting/model selection. |

**Pilot first, then expand only within the approved envelope.** The 26 trials
are not a required quota regardless of safety/data quality. Stop on the reviewed
force/torque/depth/time conditions, fixture movement, peg slip or other operator
stop conditions. Do not increase force to overcome a jam or disable safeguards.

For each attempt label contact onset, depth reached, insertion + dwell, jam,
slip/drop, timeout/abort, fixture movement and withdrawal. Store peak/trace
axial/lateral forces and moments only when the wrench is qualified. Define full
insertion from measured peg/hole geometry and relative depth, not TCP travel alone.

Insertion loads combine geometry, material friction, controller compliance,
finger/peg/fixture deformation and multiple contacts. They do not uniquely identify
all physical parameters. Conditional material-pair sliding or local force–deflection
tests may be needed. Do not call effective compliance a measured material modulus.

## File contract per trial

| Save | Source |
|---|---|
| Native command and feedback logs | Real collection program, driver and robot; keep requested and final dispatched commands distinct. |
| Gripper commands and measured opening | Actual gripper controller/driver; native current/effort/status if available. |
| Raw wrist sensor and processed wrench, separately | Installed SDK/driver exports with explicit mapping, frames/origins, units, compensation and times. |
| Normal/tangential reference force or force map | Independent calibrated instrument or validated setup-specific map, if present. |
| Original video + frame timestamps | Fixed camera; trial ID, sync event and camera calibration where quantitative. |
| Relative pose/depth/deflection measurements | Calibrated video tracking or other independent measurement method, with uncertainty. |
| Setup + trial manifest | Exact object/hole, controller, sensor, gripper, program revision, timestamps, split, numeric condition, outcome and any abort. |

Suggested normalized filenames are specified in `signal_contract.json`; preserve
native recordings first. No contact evidence parser is installed yet. Missing
channels stay unavailable; no fabricated zeros, timestamps or completed statuses.

Flexiv names depend on SDK version. Older documented APIs use `ft_sensor_raw` and
`ext_wrench_in_tcp`; RDK 2.1 uses `raw_ft_sensor` and `tcp_wrench_local`. A processed
external-wrench estimate is not the same channel as a raw physical F/T reading.
The RDK 2.1 docs explicitly say raw F/T is zero when no sensor is installed.
The engineer must verify the installed API and ROS shim, not blindly rename fields.
[Flexiv RDK 2.1 states](https://www.flexiv.com/software/rdk/api/v2.1/structflexiv_1_1rdk_1_1_robot_states.html)
· [Earlier state fields](https://www.flexiv.com/software/rdk/api/structflexiv_1_1rdk_1_1_robot_states.html)

Recheck no-contact wrench with the actual held peg/tool configuration before
insertion. Record all tare and gravity/payload compensation settings. Keep raw
measurements; any transformation/filtering/compensation is a separate derived
record. Never apply compensation twice or treat logger frequency as sensor bandwidth.

## Create the recipe-guided collection kit

```bash
newton-calibration guide recipes
newton-calibration guide start \
  --asset /absolute/path/to/Flexiv_Rizon4s_Grav.usd \
  --goal "Collect real grasp evidence for insertion transfer" \
  --recipe grasp_contact@1 --session runs/flexiv-grasp
newton-calibration guide run --session runs/flexiv-grasp

newton-calibration guide start \
  --asset /absolute/path/to/Flexiv_Rizon4s_Grav.usd \
  --goal "Collect real peg insertion evidence" \
  --recipe peg_insertion@1 --session runs/flexiv-insertion
newton-calibration guide run --session runs/flexiv-insertion
```

Aliases `grasp` and `insertion` resolve to those versioned IDs. Each session produces:

- `LAB_CHECKLIST.html`: printable customer/technician checklist.
- `recipe.snapshot.json` and `collection_plan.json`: procedure and scope.
- `setup.to_review.json`: unknown controller/tool/sensor/numeric conditions.
- `signal_contract.json`: measurements and their qualification limits.
- `run_sheet.template.csv` and `trial_templates/`: not-collected trial records.
- `bundle.json`: hashes binding the generated files to the session and recipe.

Copy the blank templates into a **separate recording directory** to complete them.
Do not edit the immutable proposal bundle. To update setup, copy and complete the
answers file, then use `guide provide --answers ... --source ...` and `guide run`;
this creates a new revision and preserves the old proposal. Answer sections are
replaced, not implicitly deep-merged: retain the facts you still want to supply.
Unknowns may remain null. The supplied answers are declarations, not verified
sensor readings, calibrated force maps or hardware permission.

`collection_spec_prepared` means the specification exists. It does not mean
scientific analysis/plan or fitting has run. `--execute` cannot bypass this boundary.
Preview is explicitly `not_supported`, not passed or silently omitted. No arm-PD
motions are substituted for an insertion controller. There is no hardware driver
or dynamic code import from a recipe.

## How later fitting will use this evidence

Record upstream configuration now, even if MVP1/MVP2 fitting has not finished.
Future staged fitting must verify the upstream packages (or separately qualify
their response), lock the controller/runtime and fitting split, then fit only
observable parameters. Backend-specific Newton contact fields need an installed,
tested binding; material parameters do not behave identically across all solvers.
[Newton solver documentation](https://github.com/newton-physics/newton/blob/main/docs/solvers/index.rst)

Held-out episodes assess predictions; separate real-policy evaluation is needed
for transfer. Match controller, policy, task criteria and operating conditions.
Do not use the same frames/trials both to fit and claim independent validation.

The experiment families reuse the earlier local Flexiv collection study's B0–B5
and I0–I5/H1–H3 structure. These are new versioned product collection contracts;
old template folders and schematics remain preserved and are not treated as real data.
