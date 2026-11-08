# Kaggle notebook: aqua20-demo, cell 2 (run interactively; prints a public gradio.live link)
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== DEMO CELL 2: Gradio web app with a public link (keeps running until you stop the cell) =====
import gradio as gr
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Grad-CAM on the highest-weighted ConvNeXt network (CNN feature maps give clean heat-maps)
cnx = max([i for i, r in enumerate(POOL) if r.startswith("convnext")], key=lambda i: WEIGHTS[i])
CAM_MODEL, CAM_TF = MODELS[cnx]; CAM_IS_CLAHE = "_clahe_" in POOL[cnx]
CAM = GradCAM(model=CAM_MODEL, target_layers=[CAM_MODEL.stages[-1]])
print("Grad-CAM network:", POOL[cnx])

def classify(img):
    if img is None:
        return None, None, None, "Please upload an image."
    p, orig, enh = ensemble_probs(img)
    top = int(p.argmax()); conf = float(p[top])
    src = enh if CAM_IS_CLAHE else orig
    x = CAM_TF(Image.fromarray(src)).unsqueeze(0).to(dev)
    g = CAM(input_tensor=x, targets=[ClassifierOutputTarget(top)])[0]
    vis = show_cam_on_image(cv2.resize(orig, (224, 224)).astype(np.float32) / 255.0, g, use_rgb=True)
    if conf >= 0.80:   note = f"High confidence ({conf:.0%}). The model is calibrated (ECE 0.02), so this is reliable."
    elif conf >= 0.50: note = f"Medium confidence ({conf:.0%}). Check the top-5 list."
    else:              note = f"Low confidence ({conf:.0%}). Send this image to an expert for review."
    return {NAMES[k]: float(p[k]) for k in np.argsort(-p)[:5]}, enh, vis, note

# Example images: one test image from each of 8 classes
want = ["fish", "coral", "turtle", "jellyfish", "shark", "starfish", "eel", "octopus"]
EX = [[f"{DATA_BASE}/{te[te['class'] == c].iloc[0]['path_orig']}"] for c in want if (te["class"] == c).any()]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="numpy", label="Upload an underwater image"),
    outputs=[gr.Label(num_top_classes=5, label="Prediction (top-5)"),
             gr.Image(label="CLAHE-enhanced image (LAB, L channel)"),
             gr.Image(label="Grad-CAM: where the model looked"),
             gr.Textbox(label="Confidence")],
    examples=EX,
    title="CLAHE-Enhanced Underwater Image Classification (AQUA20, 20 classes)",
    description="Proposed model: calibrated weighted ensemble of ConvNeXt-Small and EVA-02-Base trained on original and "
                "CLAHE images (12 networks). AQUA20 test accuracy 92.80%, macro-F1 0.897.")
demo.launch(share=True, debug=True)
