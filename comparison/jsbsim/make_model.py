"""Build a controlled copy of JSBSim's Cessna 172 for the comparison.

The derived XML is generated at runtime from the installed JSBSim package. It
is intentionally not committed because the source aircraft data has its own
license. See comparison/jsbsim/README.md for the comparison scope and caveats.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import jsbsim

MODEL_NAME = "c172x_direct"
MODEL_NAME_NO_ADOT = "c172x_direct_noadot"
BUILD_DIR = Path(__file__).resolve().parent / "build"

# Symmetric payload: lb, x/y/z inches. Symmetry removes lateral payload asymmetry.
PAYLOAD = [
    ("PILOT", 165.0, 36.0, -14.0, 24.0),
    ("CO-PILOT", 165.0, 36.0, 14.0, 24.0),
    ("PASSENGER 1", 118.0, 60.0, -14.0, 24.0),
    ("PASSENGER 2", 118.0, 60.0, 14.0, 24.0),
    ("LUGGAGE", 120.0, 95.0, 0.0, 24.0),
]

DIRECT_FCS = """
<flight_control name="direct">
  <channel name="direct surfaces">
    <pure_gain name="elevator">
      <input>fcs/elevator-cmd-norm</input><gain>1.0</gain><output>fcs/elevator-pos-rad</output>
    </pure_gain>
    <pure_gain name="aileron">
      <input>fcs/aileron-cmd-norm</input><gain>1.0</gain><output>fcs/effective-aileron-pos</output>
    </pure_gain>
    <pure_gain name="rudder">
      <input>fcs/rudder-cmd-norm</input><gain>1.0</gain><output>fcs/rudder-pos-rad</output>
    </pure_gain>
    <pure_gain name="flaps">
      <input>fcs/flap-cmd-norm</input><gain>0.0</gain><output>fcs/flap-pos-deg</output>
    </pure_gain>
  </channel>
</flight_control>
"""


def source_aircraft_file() -> Path:
    return Path(jsbsim.__file__).resolve().parent / "aircraft" / "c172x" / "c172x.xml"


def _thrust_block(cg_in: tuple[float, float, float]) -> ET.Element:
    xml = f"""
<external_reactions>
  <force name="thrust" frame="BODY" unit="LBS">
    <location unit="IN"><x>{cg_in[0]}</x><y>{cg_in[1]}</y><z>{cg_in[2]}</z></location>
    <direction><x>1</x><y>0</y><z>0</z></direction>
  </force>
</external_reactions>
"""
    return ET.fromstring(xml)


def _zero_alpha_dot_terms(root: ET.Element) -> int:
    changed = 0
    for fn in root.iter("function"):
        if fn.get("name") in ("aero/coefficient/CLadot", "aero/coefficient/Cmadot"):
            value = fn.find("product/value")
            if value is None:
                raise RuntimeError(f"unexpected JSBSim XML layout for {fn.get('name')}")
            value.text = "0.0"
            changed += 1
    return changed


def build_model(no_alpha_dot: bool = False) -> Path:
    """Write the derived model and return the JSBSim root directory.

    no_alpha_dot is a diagnostic variant because the repository has no alpha-dot
    lift/pitch terms. It helps isolate this structural source of modal mismatch.
    """
    src = ET.parse(source_aircraft_file()).getroot()
    root = copy.deepcopy(src)
    root.set("name", "Cessna C-172 (direct controls, fixed thrust)")

    for tag in ("propulsion", "system", "autopilot", "flight_control", "output", "input"):
        for el in root.findall(tag):
            root.remove(el)

    mass = root.find("mass_balance")
    if mass is None:
        raise RuntimeError("JSBSim C172 model has no mass_balance element")
    for pm in mass.findall("pointmass"):
        mass.remove(pm)
    for name, lb, x, y, z in PAYLOAD:
        pm = ET.SubElement(mass, "pointmass", name=name)
        weight = ET.SubElement(pm, "weight", unit="LBS")
        weight.text = str(lb)
        location = ET.SubElement(pm, "location", unit="IN")
        for axis, value in zip("xyz", (x, y, z)):
            ET.SubElement(location, axis).text = str(value)

    cg = mass.find("location[@name='CG']")
    if cg is None or any(cg.find(axis) is None for axis in "xyz"):
        raise RuntimeError("JSBSim C172 model has no parseable CG location")
    cg_in = tuple(float(cg.find(axis).text) for axis in "xyz")

    aero = root.find("aerodynamics")
    if aero is None:
        raise RuntimeError("JSBSim C172 model has no aerodynamics section")
    idx = list(root).index(aero)
    root.insert(idx, ET.fromstring(DIRECT_FCS))
    root.insert(idx, _thrust_block(cg_in))

    name = MODEL_NAME_NO_ADOT if no_alpha_dot else MODEL_NAME
    if no_alpha_dot and _zero_alpha_dot_terms(root) != 2:
        raise RuntimeError("expected JSBSim C172 CLadot and Cmadot terms")
    out_dir = BUILD_DIR / "aircraft" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_dir / f"{name}.xml", encoding="utf-8", xml_declaration=True)
    return BUILD_DIR


def new_fdm(dt_s: float = 0.01, no_alpha_dot: bool = False) -> "jsbsim.FGFDMExec":
    """Create a JSBSim executive with the controlled aircraft loaded."""
    root = build_model(no_alpha_dot)
    fdm = jsbsim.FGFDMExec(str(root))
    fdm.set_debug_level(0)
    fdm.set_aircraft_path("aircraft")
    fdm.set_engine_path("engine")
    fdm.set_systems_path("systems")
    model = MODEL_NAME_NO_ADOT if no_alpha_dot else MODEL_NAME
    if not fdm.load_model(model):
        raise RuntimeError(f"JSBSim could not load derived aircraft {model!r}")
    fdm.set_dt(dt_s)
    return fdm
