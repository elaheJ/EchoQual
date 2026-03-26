"""
Canonical text descriptions for standard echocardiographic views.

These serve as "reference anchors" in the vision-language quality assessment:
high-quality images should have high cosine similarity to canonical descriptions
of their expected view, while poor-quality images will diverge.

Covers five standard views:
  1. PLAX         — Parasternal Long Axis
  2. A3C          — Apical 3-Chamber
  3. A4C          — Apical 4-Chamber
  4. Doppler_A3C  — Doppler Apical 3-Chamber (Aortic Valve flow)
  5. Doppler_PLAX — Doppler Parasternal Long Axis (IVS flow)
"""

# -------------------------------------------------------------------------
# 1. Parasternal Long Axis (PLAX)
# -------------------------------------------------------------------------
PLAX_CANONICAL = [
    "Standard parasternal long-axis view clearly showing the left ventricle, "
    "aortic root, aortic valve, mitral valve, and left atrium with appropriate "
    "alignment of the interventricular septum and posterior wall.",

    "High-quality PLAX view with the aortic root and left atrium well-defined, "
    "clear visualization of mitral and aortic valve leaflets, and adequate "
    "depth setting showing the descending aorta posterior to the left atrium.",

    "Properly acquired parasternal long-axis echocardiogram demonstrating "
    "the left ventricle from base to mid-cavity, open and closing aortic "
    "valve cusps, and continuous endocardial borders of the septum and "
    "posterior wall.",

    "Optimal PLAX view with the ultrasound beam perpendicular to the "
    "interventricular septum, clear delineation of the right ventricular "
    "outflow tract anteriorly, and the mitral valve apparatus fully visible.",

    "Well-positioned parasternal long-axis view with balanced gain settings, "
    "no reverberation artifacts, and complete visualization of the left "
    "ventricular outflow tract contiguous with the aortic root.",
]

PLAX_POOR_QUALITY = [
    "Suboptimal parasternal long-axis view with oblique cut through the "
    "left ventricle, poor visualization of the aortic valve, and excessive "
    "near-field reverberation artifacts.",

    "Off-axis PLAX with foreshortened left ventricle, incomplete aortic root "
    "visualization, and excessive gain producing blooming artifacts.",

    "Technically limited parasternal long-axis view with acoustic shadowing "
    "from ribs, dropout of the posterior wall, and inability to clearly "
    "identify the mitral valve leaflets.",

    "Low-quality PLAX with the interventricular septum not aligned "
    "perpendicular to the beam, resulting in apparent septal thickening "
    "artifact and poor tissue-blood contrast.",

    "Poorly acquired parasternal view with significant respiratory motion, "
    "inconsistent gain, and missing visualization of the left atrium "
    "posterior to the aortic root.",
]

# -------------------------------------------------------------------------
# 2. Apical 3-Chamber (A3C)
# -------------------------------------------------------------------------
A3C_CANONICAL = [
    "Standard apical three-chamber view showing the left atrium, left "
    "ventricle, and aortic outflow tract with clear visualization of both "
    "mitral and aortic valves in the same imaging plane.",

    "High-quality apical three-chamber echocardiogram with the left "
    "ventricular apex at the top of the sector, well-defined endocardial "
    "borders of the anteroseptal and inferolateral walls, and the aortic "
    "root clearly visualized.",

    "Properly acquired A3C view demonstrating unforeshortened left ventricle, "
    "clear aortic valve opening and closure, and good alignment of the left "
    "ventricular outflow tract for hemodynamic assessment.",

    "Optimal apical three-chamber view with symmetric ventricular walls, "
    "complete visualization from apex to aortic annulus, and adequate frame "
    "rate for wall motion analysis.",

    "Well-positioned apical long-axis view with the transducer at the true "
    "cardiac apex, showing the left atrium, mitral valve, left ventricle, "
    "and aortic valve in a single sweep without off-axis artifacts.",
]

A3C_POOR_QUALITY = [
    "Suboptimal apical three-chamber view with foreshortened left ventricle, "
    "poor visualization of the aortic outflow tract, and significant "
    "near-field clutter obscuring the apex.",

    "Off-axis A3C with unclear aortic valve, incomplete left atrial "
    "visualization, and excessive lateral wall dropout.",

    "Technically limited apical three-chamber study with lung interference, "
    "poor endocardial definition, and inability to visualize the aortic "
    "root continuously.",

    "Low-quality A3C with the transducer mispositioned laterally, resulting "
    "in a hybrid view between two-chamber and three-chamber orientations.",

    "Poorly acquired apical view with respiratory motion degrading wall "
    "motion visualization and poor tissue harmonic penetration.",
]

# -------------------------------------------------------------------------
# 3. Apical 4-Chamber (A4C)
# -------------------------------------------------------------------------
A4C_CANONICAL = [
    "Standard apical four-chamber echocardiographic view with clear "
    "visualization of all four cardiac chambers, intact interventricular "
    "and interatrial septa, and well-defined mitral and tricuspid valve "
    "leaflets.",

    "High-quality apical four-chamber view showing the left ventricle, "
    "right ventricle, left atrium, and right atrium with good endocardial "
    "border definition and appropriate sector width and depth settings.",

    "Properly acquired apical four-chamber echocardiogram demonstrating "
    "symmetric alignment of the cardiac apex at the top of the sector, "
    "with the interventricular septum oriented vertically and all four "
    "chambers clearly delineated.",

    "Optimal A4C view with complete visualization of the left and right "
    "ventricles from apex to base, clear atrioventricular valve motion, "
    "and adequate near-field and far-field resolution without excessive "
    "gain artifacts.",

    "Well-positioned apical four-chamber view with the transducer at the "
    "cardiac apex, showing balanced chamber sizes, clear tissue-blood "
    "interfaces, and adequate frame rate for functional assessment.",
]

A4C_POOR_QUALITY = [
    "Suboptimal apical view with foreshortened left ventricle, poor "
    "endocardial border definition, and excessive acoustic shadowing.",

    "Off-axis echocardiographic view with unclear chamber boundaries, "
    "missing visualization of one or more cardiac chambers, and excessive "
    "gain.",

    "Low-quality ultrasound image with significant speckle noise, dropout "
    "artifacts, and inability to identify standard cardiac structures.",

    "Poorly acquired echocardiogram with the transducer misaligned from "
    "the cardiac apex, resulting in oblique cross-section and "
    "foreshortening.",

    "Technically limited study with near-field reverberation artifacts, "
    "incomplete visualization of the left ventricular apex, and poor "
    "lateral resolution.",
]

# -------------------------------------------------------------------------
# 4. Doppler Apical 3-Chamber (Aortic Valve flow)
# -------------------------------------------------------------------------
DOPPLER_A3C_CANONICAL = [
    "Color Doppler apical three-chamber view with a well-positioned color "
    "box over the aortic valve showing laminar systolic flow through the "
    "left ventricular outflow tract and across the aortic valve.",

    "High-quality Doppler A3C demonstrating normal antegrade flow across "
    "the aortic valve with no aliasing at appropriate Nyquist velocity, "
    "and clear delineation of the vena contracta.",

    "Properly acquired color-flow Doppler in the apical three-chamber "
    "orientation with the interrogation angle aligned parallel to aortic "
    "outflow, minimal color bleeding outside the vessel lumen, and adequate "
    "temporal resolution.",

    "Optimal Doppler apical long-axis view with the color sector focused "
    "on the aortic valve region, appropriate scale settings to detect both "
    "stenotic and regurgitant jets, and clear distinction between systolic "
    "and diastolic flow patterns.",

    "Well-acquired color Doppler A3C with the sample volume correctly "
    "placed at the aortic annulus level, laminar flow demonstrated in "
    "the LVOT, and any regurgitant jet clearly visible in the left "
    "ventricular cavity during diastole.",
]

DOPPLER_A3C_POOR_QUALITY = [
    "Suboptimal Doppler A3C with the color box misaligned from the aortic "
    "valve, excessive color bleeding into adjacent myocardium, and "
    "inadequate Nyquist velocity resulting in aliasing artifact.",

    "Low-quality color Doppler apical three-chamber with poor angle of "
    "incidence to flow, wall motion artifact contaminating the color map, "
    "and inability to distinguish stenotic from regurgitant jets.",

    "Technically limited Doppler A3C with too large a color sector reducing "
    "temporal resolution, excessive gain causing flash artifact, and poor "
    "frame rate.",

    "Poorly acquired Doppler study with color scale set too low causing "
    "widespread aliasing, poor tissue suppression, and incomplete coverage "
    "of the aortic valve plane.",

    "Off-axis Doppler apical view with the interrogation beam not aligned "
    "with flow direction, underestimating true velocities and missing "
    "eccentric regurgitant jets.",
]

# -------------------------------------------------------------------------
# 5. Doppler Parasternal Long Axis (IVS flow)
# -------------------------------------------------------------------------
DOPPLER_PLAX_CANONICAL = [
    "Color Doppler parasternal long-axis view with a focused color box "
    "over the interventricular septum and left ventricular outflow tract, "
    "demonstrating laminar flow through the LVOT without turbulence.",

    "High-quality Doppler PLAX showing flow across the interventricular "
    "septum with appropriate Nyquist limit, clear identification of any "
    "ventricular septal defect jets, and well-suppressed tissue motion "
    "artifact.",

    "Properly acquired color-flow PLAX with the color sector positioned "
    "to assess flow across the interventricular septum, aortic valve, "
    "and mitral valve regions simultaneously with adequate spatial "
    "resolution.",

    "Optimal Doppler parasternal long-axis view detecting normal "
    "diastolic inflow through the mitral valve and systolic outflow "
    "through the aortic valve, with any abnormal trans-septal flow "
    "clearly visible.",

    "Well-positioned color Doppler PLAX with minimal wall filter "
    "artifact, appropriate color gain, and clear demonstration of "
    "flow convergence zones if present near the interventricular septum.",
]

DOPPLER_PLAX_POOR_QUALITY = [
    "Suboptimal Doppler PLAX with the color box too large, reducing "
    "frame rate and spatial resolution, and excessive flash artifact "
    "from cardiac translation.",

    "Low-quality color Doppler parasternal view with poor tissue "
    "suppression causing motion artifact over the septum, inappropriate "
    "color scale, and inability to detect low-velocity septal flow.",

    "Technically limited Doppler PLAX with the color sector not covering "
    "the interventricular septum adequately, poor angle of insonation, "
    "and color bleeding into pericardial space.",

    "Poorly acquired Doppler parasternal view with excessive gain causing "
    "noise throughout the color map, inability to distinguish true flow "
    "from artifact, and inadequate Nyquist velocity.",

    "Off-axis Doppler PLAX with the beam perpendicular to flow direction "
    "across the septum, severely underestimating flow velocities and "
    "missing small septal defects.",
]

# -------------------------------------------------------------------------
# Additional standard views (for future expansion)
# -------------------------------------------------------------------------
PSAX_CANONICAL = [
    "Standard parasternal short-axis view at the mid-papillary muscle level "
    "showing a circular cross-section of the left ventricle with symmetric "
    "wall thickness and clear endocardial borders.",

    "Well-acquired PSAX view demonstrating the left ventricle as a circular "
    "structure with papillary muscles visible at approximately the 4 and 8 "
    "o'clock positions and uniform myocardial echogenicity.",
]

A2C_CANONICAL = [
    "Standard apical two-chamber view showing the left ventricle and left "
    "atrium with the mitral valve in between, clear anterior and inferior "
    "wall visualization, and the left atrial appendage visible.",

    "Properly aligned A2C view with the left ventricle centered, good "
    "endocardial definition of anterior and inferior walls, and adequate "
    "depth to include the entire left atrium.",
]

SUBCOSTAL_CANONICAL = [
    "Standard subcostal four-chamber view with all four cardiac chambers "
    "visible, the liver serving as an acoustic window, and adequate "
    "visualization of the interatrial septum.",
]

# -------------------------------------------------------------------------
# Aggregate all canonical texts by view
# -------------------------------------------------------------------------
CANONICAL_TEXTS = {
    "PLAX":              PLAX_CANONICAL,
    "PLAX_poor":         PLAX_POOR_QUALITY,
    "A3C":               A3C_CANONICAL,
    "A3C_poor":          A3C_POOR_QUALITY,
    "A4C":               A4C_CANONICAL,
    "A4C_poor":          A4C_POOR_QUALITY,
    "Doppler_A3C":       DOPPLER_A3C_CANONICAL,
    "Doppler_A3C_poor":  DOPPLER_A3C_POOR_QUALITY,
    "Doppler_PLAX":      DOPPLER_PLAX_CANONICAL,
    "Doppler_PLAX_poor": DOPPLER_PLAX_POOR_QUALITY,
    "PSAX":              PSAX_CANONICAL,
    "A2C":               A2C_CANONICAL,
    "SUBCOSTAL":         SUBCOSTAL_CANONICAL,
}

# View label mapping (for classification head)
VIEW_LABELS = {
    "PLAX": 0,
    "A3C": 1,
    "A4C": 2,
    "Doppler_A3C": 3,
    "Doppler_PLAX": 4,
}

NUM_VIEWS = len(VIEW_LABELS)


def get_canonical_texts(view: str = "A4C", include_poor: bool = True) -> dict:
    """
    Get canonical text descriptions for a view.

    Args:
        view: View name (PLAX, A3C, A4C, Doppler_A3C, Doppler_PLAX, etc.)
        include_poor: Whether to include poor-quality descriptions

    Returns:
        dict with 'good' and optionally 'poor' text lists
    """
    result = {"good": CANONICAL_TEXTS.get(view, A4C_CANONICAL)}
    if include_poor:
        poor_key = f"{view}_poor"
        result["poor"] = CANONICAL_TEXTS.get(poor_key, A4C_POOR_QUALITY)
    return result


def get_all_view_texts(include_poor: bool = True) -> dict:
    """
    Get canonical texts for all five standard views.

    Returns:
        dict mapping view name to {'good': [...], 'poor': [...]}
    """
    return {view: get_canonical_texts(view, include_poor) for view in VIEW_LABELS}
