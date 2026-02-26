"""
Canonical text descriptions for standard echocardiographic views.

These serve as "reference anchors" in the vision-language quality assessment:
high-quality images should have high cosine similarity to canonical descriptions
of their expected view, while poor-quality images will diverge.

For EchoNet-Dynamic (all A4C view), we provide multiple paraphrases of a
well-acquired apical four-chamber view. For future multi-view extension,
descriptions for all standard views are included.
"""

# Apical Four-Chamber (A4C) — the view in EchoNet-Dynamic
A4C_CANONICAL = [
    "Standard apical four-chamber echocardiographic view with clear visualization "
    "of all four cardiac chambers, intact interventricular and interatrial septa, "
    "and well-defined mitral and tricuspid valve leaflets.",

    "High-quality apical four-chamber view showing the left ventricle, right ventricle, "
    "left atrium, and right atrium with good endocardial border definition and "
    "appropriate sector width and depth settings.",

    "Properly acquired apical four-chamber echocardiogram demonstrating symmetric "
    "alignment of the cardiac apex at the top of the sector, with the interventricular "
    "septum oriented vertically and all four chambers clearly delineated.",

    "Optimal A4C view with complete visualization of the left and right ventricles "
    "from apex to base, clear atrioventricular valve motion, and adequate "
    "near-field and far-field resolution without excessive gain artifacts.",

    "Well-positioned apical four-chamber view with the transducer at the cardiac "
    "apex, showing balanced chamber sizes, clear tissue-blood interfaces, and "
    "adequate frame rate for functional assessment.",
]

# Descriptions of POOR quality A4C views (used as negative anchors)
A4C_POOR_QUALITY = [
    "Suboptimal apical view with foreshortened left ventricle, "
    "poor endocardial border definition, and excessive acoustic shadowing.",

    "Off-axis echocardiographic view with unclear chamber boundaries, "
    "missing visualization of one or more cardiac chambers, and excessive gain.",

    "Low-quality ultrasound image with significant speckle noise, dropout "
    "artifacts, and inability to identify standard cardiac structures.",

    "Poorly acquired echocardiogram with the transducer misaligned from the "
    "cardiac apex, resulting in oblique cross-section and foreshortening.",

    "Technically limited study with near-field reverberation artifacts, "
    "incomplete visualization of the left ventricular apex, and poor "
    "lateral resolution.",
]

# Additional standard views for future multi-view extension
PLAX_CANONICAL = [
    "Standard parasternal long-axis view clearly showing the left ventricle, "
    "aortic root, aortic valve, mitral valve, and left atrium with appropriate "
    "alignment of the interventricular septum and posterior wall.",

    "High-quality PLAX view with the aortic root and left atrium well-defined, "
    "clear visualization of mitral and aortic valve leaflets, and adequate "
    "depth setting showing the descending aorta posterior to the left atrium.",
]

PSAX_CANONICAL = [
    "Standard parasternal short-axis view at the mid-papillary muscle level "
    "showing a circular cross-section of the left ventricle with symmetric "
    "wall thickness and clear endocardial borders.",

    "Well-acquired PSAX view demonstrating the left ventricle as a circular "
    "structure with papillary muscles visible at approximately the 4 and 8 "
    "o'clock positions and uniform myocardial echogenicity.",
]

A2C_CANONICAL = [
    "Standard apical two-chamber view showing the left ventricle and left atrium "
    "with the mitral valve in between, clear anterior and inferior wall "
    "visualization, and the left atrial appendage visible.",

    "Properly aligned A2C view with the left ventricle centered, good "
    "endocardial definition of anterior and inferior walls, and adequate "
    "depth to include the entire left atrium.",
]

SUBCOSTAL_CANONICAL = [
    "Standard subcostal four-chamber view with all four cardiac chambers "
    "visible, the liver serving as an acoustic window, and adequate "
    "visualization of the interatrial septum.",
]

# Aggregate all canonical texts by view
CANONICAL_TEXTS = {
    "A4C": A4C_CANONICAL,
    "A4C_poor": A4C_POOR_QUALITY,
    "PLAX": PLAX_CANONICAL,
    "PSAX": PSAX_CANONICAL,
    "A2C": A2C_CANONICAL,
    "SUBCOSTAL": SUBCOSTAL_CANONICAL,
}


def get_canonical_texts(view: str = "A4C", include_poor: bool = True) -> dict:
    """
    Get canonical text descriptions for a view.

    Returns:
        dict with 'good' and optionally 'poor' text lists
    """
    result = {"good": CANONICAL_TEXTS.get(view, A4C_CANONICAL)}
    if include_poor:
        result["poor"] = CANONICAL_TEXTS.get(f"{view}_poor", A4C_POOR_QUALITY)
    return result
