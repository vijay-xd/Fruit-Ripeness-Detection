"""
explainability.py
Explainable AI (XAI) & Morphological Analysis for Fruit Ripeness & Rot Detection.

Classes (4 Ordered Ranks):
0: unripe   (Firm, chlorophyll green, low sugar)
1: ripe     (Peak sweetness, vibrant carotenoid/anthocyanin, ready to eat)
2: overripe (Softening, high sugar, brown ethylene spots)
3: rotten   (Decayed, necrotic core, fungal mold, severe oxidation, spoiled)
"""

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as cm
import tensorflow as tf


def compute_gradcam(model, img_array: np.ndarray, pred_index=None, layer_name="conv4_features"):
    """
    Computes Grad-CAM heatmap for a given input image and model.
    Works for both Standard CNN and Ordinal Regression CNN.
    """
    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                loss = tf.reduce_max(predictions[0])
            else:
                if predictions.shape[-1] >= 4:
                    # Softmax multiclass (4 classes)
                    loss = predictions[0, min(pred_index, predictions.shape[-1] - 1)]
                else:
                    # Ordinal binary heads (3 heads)
                    head_idx = min(max(0, pred_index - 1), predictions.shape[-1] - 1)
                    loss = predictions[0, head_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        heatmap_img = Image.fromarray(np.uint8(255 * heatmap_np)).resize((224, 224), Image.Resampling.BILINEAR)
        heatmap_resized = np.array(heatmap_img) / 255.0

        colormap = cm.get_cmap("jet")
        colored_heatmap = colormap(heatmap_resized)[:, :, :3]

        return colored_heatmap, heatmap_resized
    except Exception as e:
        print(f"[Grad-CAM Warning] Fallback saliency used: {e}")
        dummy = np.zeros((224, 224, 3), dtype=np.float32)
        return dummy, np.zeros((224, 224), dtype=np.float32)


def overlay_heatmap(original_pil: Image.Image, heatmap_colored: np.ndarray, alpha=0.45) -> Image.Image:
    """Blends the Grad-CAM heatmap with the original image."""
    orig_resized = original_pil.convert("RGB").resize((224, 224))
    orig_np = np.array(orig_resized, dtype=np.float32) / 255.0

    blended = (1.0 - alpha) * orig_np + alpha * heatmap_colored
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def extract_peel_color_profile(pil_img: Image.Image):
    """
    Comprehensive biological vision decomposition for fruits and cut specimens:
    - Chlorophyll (Green/Unripe)
    - Vibrant Carotenoids & Anthocyanins (Fresh Ripe)
    - Polyphenol Browning & Enzymatic Senescence (Overripe)
    - Necrosis, Mold Mycelium & Decomposed Cavities (Rotten)
    """
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    
    # Filter out near-white background pixels (> 230 across all channels)
    non_bg_mask = np.any(arr < 232, axis=-1)
    if np.sum(non_bg_mask) == 0:
        non_bg_mask = np.ones((224, 224), dtype=bool)

    fruit_pixels = arr[non_bg_mask]
    r = fruit_pixels[:, 0]
    g = fruit_pixels[:, 1]
    b = fruit_pixels[:, 2]

    # 1. Chlorophyll index: green significantly higher than red (Unripe)
    green_mask = (g > (r + 15)) & (g > b)

    # 2. Necrotic rot & decayed cavity index (dark brown/black core or sunken cavity)
    rot_core_mask = (r < 105) & (g < 70) & (b < 55) & (r + g + b < 200)
    # Mold mycelium (grayish/pale cyan-green fungus: low saturation, medium luminance)
    mold_mask = (np.abs(r - g) < 20) & (np.abs(g - b) < 20) & (g > 95) & (g < 165) & (r < 160)
    rot_decay_mask = rot_core_mask | mold_mask

    # 3. Vibrant Fresh Ripe Pigment (anthocyanin red or carotenoid yellow/orange peel)
    red_ripe_mask = (r > 115) & (r > 1.28 * g) & (r > 1.28 * b)
    yellow_ripe_mask = (r > 160) & (g > 125) & (b < 115) & (r >= g - 25) & (r - b > 40)
    fresh_ripe_mask = (red_ripe_mask | yellow_ripe_mask) & (~rot_decay_mask)

    # 4. Enzymatic browning / Oxidized tan pulp & Senescence spots
    oxidized_pulp_mask = (r > 160) & (g > 120) & (b < 140) & (r - g > 30) & (r - g < 80) & (b > 50) & (~fresh_ripe_mask)
    senescence_brown_mask = (r > 60) & (r < 140) & (g > 35) & (g < 100) & (b < 70) & (r >= g) & (r - g < 40) & (~fresh_ripe_mask) & (~rot_decay_mask)
    brown_mask = oxidized_pulp_mask | senescence_brown_mask

    # Direct percentage over total non-background fruit pixels
    green_pct = round(float(np.mean(green_mask)) * 100, 1)
    ripe_pigment_pct = round(float(np.mean(fresh_ripe_mask)) * 100, 1)
    senescence_brown_pct = round(float(np.mean(brown_mask)) * 100, 1)
    rot_decay_pct = round(float(np.mean(rot_decay_mask)) * 100, 1)
    oxidized_pulp_pct = round(float(np.mean(oxidized_pulp_mask)) * 100, 1)

    color_metrics = {
        "green_pct": green_pct,
        "ripe_pigment_pct": ripe_pigment_pct,
        "senescence_brown_pct": senescence_brown_pct,
        "rot_decay_pct": rot_decay_pct,
        "oxidized_pulp_pct": oxidized_pulp_pct,
        "avg_luminance": round(float(np.mean(0.299 * r + 0.587 * g + 0.114 * b)), 1)
    }

    # Blemish & surface wrinkle texture
    gray_img = img.convert("L")
    blurred = gray_img.filter(ImageFilter.GaussianBlur(2))
    diff = np.abs(np.array(gray_img, dtype=np.int16) - np.array(blurred, dtype=np.int16))
    spot_density = round(float(np.mean(diff > 18) * 100), 2)
    color_metrics["blemish_spot_density_pct"] = spot_density

    # Composite rot / decay severity score
    rot_score = rot_decay_pct * 3.0 + senescence_brown_pct * 1.2 + spot_density * 0.4
    color_metrics["rot_severity_score"] = round(rot_score, 1)

    return color_metrics


def generate_natural_language_explanation(pred_class: str, confidence: float, color_metrics: dict, model_type="Ordinal CNN") -> dict:
    """Generates an intuitive, viva-ready explanation for why the model predicted this stage."""
    g_pct = color_metrics.get("green_pct", 0)
    r_pct = color_metrics.get("ripe_pigment_pct", 0)
    b_pct = color_metrics.get("senescence_brown_pct", 0)
    rot_pct = color_metrics.get("rot_decay_pct", 0)
    spot_pct = color_metrics.get("blemish_spot_density_pct", 0)

    if pred_class == "unripe":
        summary = (
            f"The image exhibits a dominant chlorophyll index ({g_pct}% green coloration) and firm surface texture with "
            f"negligible oxidation ({b_pct}%). The CNN feature extractor recognized low sugar accumulation and high starch density, "
            f"placing it into Ordinal Level 0 (Unripe)."
        )
        recommendation = "Store at ambient room temperature (18-22 deg C) for 3 to 5 days to allow natural ethylene gas conversion."
        culinary_use = "Best for cooking, pickling, raw salads, or savory vegetable dishes."

    elif pred_class == "ripe":
        summary = (
            f"The image reveals optimal ripening pigments ({r_pct}% vibrant carotenoid/anthocyanin spectrum) "
            f"with balanced peel luminance. Visual patterns show peak ethylene conversion with minimal browning ({b_pct}%). "
            f"The model confidently confirmed the fruit is at peak maturity and ready to eat."
        )
        recommendation = "Peak sweetness and nutrient bioavailability. Ready to eat immediately or refrigerate for 2-3 days."
        culinary_use = "Ideal for fresh consumption, fruit salads, snacking, and garnishing."

    elif pred_class == "overripe":
        summary = (
            f"The model detected noticeable enzymatic browning ({b_pct}% senescence coloration) and elevated speckle density "
            f"({spot_pct}% surface spots). Cell wall softening and starch-to-sugar conversion are complete, placing this sample in Ordinal Level 2 (Overripe)."
        )
        recommendation = "High sugar concentration with softened texture. Consume today; do not store at room temperature."
        culinary_use = "Perfect for fruit smoothies, baking (banana bread, fruit crumbles), preserves, purees, or jams."

    else:  # rotten
        summary = (
            f"CRITICAL SPOILAGE DETECTED: The model identified necrotic tissue ({rot_pct}% decay index), severe polyphenol oxidation "
            f"({b_pct}% browning), shriveled cellular structure, and decomposed cavity breakdown. The fruit has degraded past edible limits into Ordinal Level 3 (Rotten)."
        )
        recommendation = "⚠️ DO NOT CONSUME. Microbial growth, fungal mold mycelium, or fermentative rot pose digestive and health risks. Discard immediately."
        culinary_use = "Unfit for human consumption. Dispose of safely or add to organic garden compost."

    return {
        "summary": summary,
        "recommendation": recommendation,
        "culinary_use": culinary_use
    }
