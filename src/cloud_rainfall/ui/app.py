"""Streamlit web dashboard for Cloud Pattern Recognition and Rainfall Estimation."""

from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

from cloud_rainfall.config import PROJECT_ROOT, get_paths_config, get_cloud_model_config, get_rainfall_model_config
from cloud_rainfall.services.prediction_service import PredictionService
from cloud_rainfall.data.validation import validate_image_file
from cloud_rainfall.evaluation.plots import plot_probabilities_bar

st.set_page_config(
    page_title="SkyVision: Cloud Recognition & Rainfall Estimator",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_prediction_service() -> PredictionService:
    """Cache singleton prediction service across Streamlit sessions."""
    return PredictionService()


def main():
    service = get_prediction_service()
    paths_cfg = get_paths_config()
    cloud_cfg = get_cloud_model_config()
    rain_cfg = get_rainfall_model_config()

    st.sidebar.title("⛅ SkyVision System")
    st.sidebar.markdown(
        "**Machine Vision Mini Project**  \n"
        "Ground-based Sky Image Recognition & Nowcasting"
    )

    nav = st.sidebar.radio(
        "Navigation",
        [
            "Home / Overview",
            "Analyze Sky Image",
            "Model Evaluation",
            "About & Scientific Limits",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    cloud_loaded = service.cloud_engine is not None
    rain_loaded = service.rainfall_classifier is not None

    if cloud_loaded:
        st.sidebar.success("✅ Cloud CNN Active (MobileNetV3)")
    else:
        st.sidebar.error("❌ Cloud Checkpoint Missing")

    if rain_loaded:
        lbl_source = service.rainfall_metadata.get("label_source", "Proxy")
        st.sidebar.info(f"✅ Rainfall Model Active ({lbl_source})")
    else:
        st.sidebar.warning("⚠️ Rainfall Model Unavailable")

    st.sidebar.caption("Prototype Demonstration — Not an official meteorological warning system.")

    # -------------------------------------------------------------
    # SCREEN 1: Home / Overview
    # -------------------------------------------------------------
    if nav == "Home / Overview":
        st.title("⛅ Cloud Pattern Recognition from Sky Images for Rainfall Estimation")
        st.markdown(
            """
            This machine vision application analyses ground-based sky photographs using deep transfer learning
            to classify visible atmospheric cloud patterns and examine their relationship with short-term precipitation nowcasting.
            """
        )

        st.info(
            """
            **Crucial Scientific Distinction:**  
            - **Task A (Cloud Recognition):** Classifies broad sky and cloud types (e.g. clear sky, patterned altocumulus, dense dark cumulonimbus).
            - **Task B (Rainfall Estimation):** Requires timestamp-aligned rainfall measurements. Sky visual appearance provides immediate cloud optics, but true rain processes also involve moisture columns, pressure dynamics, and wind.
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Primary Vision Workflow")
            st.markdown(
                """
                1. **Image Validation**: Checks resolution, channels, extreme darkness, and washouts.
                2. **Deep CNN Inference**: Extracts high-level features via ImageNet pretrained MobileNetV3-Small.
                3. **Pattern Classification**: Computes probability distribution over standard cloud morphology classes.
                """
            )
        with col2:
            st.subheader("🌧️ Rainfall Nowcasting Mode")
            st.markdown(
                """
                1. **Bottleneck Embeddings**: Uses high-dimensional feature representations from the CNN.
                2. **Target Horizon**: Next 1 hour accumulation $[t, t+1\\text{h})$.
                3. **Scientific Integrity**: Only active when paired models are loaded. If uncalibrated or absent, estimates are honestly withheld.
                """
            )

        st.markdown("---")
        st.subheader("🚀 Quick Start")
        st.markdown(
            """
            - Switch to **'Analyze Sky Image'** in the sidebar to upload a photograph or select a sample image.
            - View **'Model Evaluation'** to inspect held-out test set metrics, precision/recall, and confusion matrices.
            """
        )

    # -------------------------------------------------------------
    # SCREEN 2 & 3: Analyze Sky Image & Results
    # -------------------------------------------------------------
    elif nav == "Analyze Sky Image":
        st.title("🔍 Sky Photograph Analysis")
        st.write("Upload a sky photograph to detect cloud patterns and estimate next-hour precipitation risk.")

        col_input, col_meta = st.columns([2, 1])

        with col_input:
            upload_mode = st.radio(
                "Select Image Source:",
                ["Upload from Computer", "Choose Demo Sky Sample"],
                horizontal=True,
            )

            image_to_analyze = None
            image_name = "Uploaded Image"

            if upload_mode == "Upload from Computer":
                uploaded_file = st.file_uploader(
                    "Choose an image (JPG, PNG, JPEG, WEBP)...",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                )
                if uploaded_file is not None:
                    image_to_analyze = Image.open(uploaded_file)
                    image_name = uploaded_file.name

            else:
                raw_dir = PROJECT_ROOT / paths_cfg["data"]["raw_dir"]
                sample_files = list(raw_dir.glob("*/*.jpg")) if raw_dir.exists() else []
                if sample_files:
                    sample_options = {f.name: f for f in sample_files[:30]}
                    selected_sample_name = st.selectbox(
                        "Select a sample image from dataset:",
                        list(sample_options.keys()),
                    )
                    image_path = sample_options[selected_sample_name]
                    image_to_analyze = Image.open(image_path)
                    image_name = selected_sample_name
                else:
                    st.info("No sample images found in data/raw. Run `python scripts/prepare_cloud_dataset.py --create-sample` to generate test images.")

        with col_meta:
            if image_to_analyze is not None:
                st.image(image_to_analyze, caption=image_name, use_container_width=True)
                val_res = validate_image_file(image_to_analyze)
                st.caption(f"Dimensions: {val_res.width} x {val_res.height} px")
                st.caption(f"Mean Intensity: {val_res.mean_intensity:.1f} / 255")

                if val_res.warnings:
                    for w in val_res.warnings:
                        st.warning(f"⚠️ {w}")

        st.markdown("---")

        # Weather context inputs (for optional multimodal evaluation)
        with st.expander("🌦️ Optional Prediction-Time Weather Context (for Multimodal Evaluation)"):
            c_w1, c_w2, c_w3, c_w4 = st.columns(4)
            with c_w1:
                t2m = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=28.0, step=0.5)
            with c_w2:
                rh2m = st.number_input("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=75.0, step=1.0)
            with c_w3:
                sp = st.number_input("Surface Pressure (hPa)", min_value=900.0, max_value=1050.0, value=1008.0, step=1.0)
            with c_w4:
                ws10 = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=14.0, step=0.5)

            weather_context = {
                "temperature_2m": t2m,
                "relative_humidity_2m": rh2m,
                "surface_pressure": sp,
                "wind_speed_10m": ws10,
            }

        if st.button("🚀 Run Vision & Nowcasting Analysis", type="primary"):
            if image_to_analyze is None:
                st.error("Please upload or select an image first.")
            else:
                with st.spinner("Processing sky image through neural network..."):
                    res = service.analyze_sky_image(image_to_analyze, weather_features=weather_context)

                if not res["success"]:
                    st.error(f"Analysis Failed: {res['error']}")
                else:
                    cloud = res["cloud_result"]
                    rain = res["rainfall_result"]

                    # Display Cloud Classification Results (Screen 3)
                    st.success("Analysis Complete!")
                    st.subheader("1. Cloud Pattern Recognition (Task A)")

                    c_res1, c_res2 = st.columns([1, 1.2])

                    with c_res1:
                        st.metric(
                            label="Predicted Cloud Category",
                            value=cloud["predicted_class"].replace("_", " ").title(),
                            delta=f"{cloud['confidence']*100:.1f}% Confidence",
                        )
                        st.markdown(f"**Meteorological Assessment:**\n{cloud['explanation']}")

                    with c_res2:
                        fig = plot_probabilities_bar(cloud["probabilities"])
                        st.pyplot(fig)

                    st.markdown("---")

                    # Display Rainfall Estimation Results (Screen 4)
                    st.subheader("2. Short-Term Rainfall Nowcasting (Task B)")

                    if not rain or not rain["is_available"]:
                        st.warning(
                            f"**Rainfall Model Unavailable**: {rain.get('message', 'No paired rainfall checkpoint found.')}"
                        )
                        st.caption("To enable nowcasting, run `python scripts/train_rainfall_model.py` after pairing images.")
                    else:
                        r_col1, r_col2, r_col3 = st.columns(3)
                        with r_col1:
                            st.metric(
                                label="Next-Hour Rain Category",
                                value=rain["predicted_category"].replace("_", " ").title(),
                            )
                        with r_col2:
                            st.metric(
                                label="Estimated Rainfall Amount",
                                value=f"{rain['predicted_mm']:.2f} mm",
                                help="Accumulated rain in next 1 hour",
                            )
                        with r_col3:
                            st.metric(
                                label="Prediction Horizon",
                                value=rain["horizon"].replace("_", " ").title(),
                            )

                        st.info(
                            f"**Model Type**: `{rain['model_type']}` | **Label Source**: `{rain['label_source']}`"
                        )

                        if rain.get("multimodal_category"):
                            st.caption(
                                f"Multimodal Context Model (Image + Current Weather) predicts: "
                                f"**{rain['multimodal_category'].replace('_', ' ').title()}**"
                            )

    # -------------------------------------------------------------
    # SCREEN 5: Model Evaluation
    # -------------------------------------------------------------
    elif nav == "Model Evaluation":
        st.title("📊 Model Evaluation & Metrics")
        st.markdown("Held-out test set performance evaluation across Vision and Nowcasting tasks.")

        tab1, tab2 = st.tabs(["Cloud Classification (Task A)", "Rainfall Nowcasting (Task B)"])

        with tab1:
            st.subheader("Cloud Classifier Test Performance")
            cm_path = PROJECT_ROOT / paths_cfg["reports"]["figures_dir"] / "cloud_confusion_matrix.png"
            pred_csv = PROJECT_ROOT / paths_cfg["reports"]["dir"] / "cloud_test_predictions.csv"

            if pred_csv.exists():
                df_preds = pd.read_csv(pred_csv)
                total_test = len(df_preds)
                acc = float(df_preds["correct"].mean())

                m1, m2, m3 = st.columns(3)
                m1.metric("Held-Out Test Samples", total_test)
                m2.metric("Test Accuracy", f"{acc*100:.2f}%")
                m3.metric("Architecture", "MobileNetV3-Small")

                if cm_path.exists():
                    st.image(str(cm_path), caption="Normalized Confusion Matrix on Test Split", width=600)

                st.subheader("Test Split Predictions Table")
                st.dataframe(df_preds[["relative_path", "class_name", "predicted_class", "confidence", "correct"]].head(20))
            else:
                st.info("No held-out test predictions found. Run `python scripts/evaluate_models.py` to evaluate.")

        with tab2:
            st.subheader("Rainfall Model Evaluation")
            scatter_path = PROJECT_ROOT / paths_cfg["reports"]["figures_dir"] / "rainfall_observed_vs_predicted.png"
            meta_path = PROJECT_ROOT / paths_cfg["artifacts"]["metadata"]

            if meta_path.exists():
                import json
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                c1, c2, c3 = st.columns(3)
                c1.metric("Image Model Macro-F1", f"{meta.get('test_macro_f1_image', 0.0):.4f}")
                c2.metric("Image Model MAE", f"{meta.get('test_mae_mm_image', 0.0):.2f} mm")
                c3.metric("Multimodal Macro-F1", f"{meta.get('test_macro_f1_multimodal', 0.0):.4f}")

                if scatter_path.exists():
                    st.image(str(scatter_path), caption="Observed vs. Predicted Next-Hour Precipitation (mm)", width=600)

                st.json(meta)
            else:
                st.info("No rainfall evaluation metadata found. Run `python scripts/train_rainfall_model.py` to train and evaluate.")

    # -------------------------------------------------------------
    # SCREEN 6: About & Limitations
    # -------------------------------------------------------------
    elif nav == "About & Scientific Limits":
        st.title("ℹ️ Scientific Foundations & Limitations")

        st.markdown(
            """
            ### Scientific Limitations of Ground Sky Photography
            1. **Atmospheric Column Bias**: A ground camera views optical cloud bottoms and scattering. Rainfall initiation requires vertical cloud thickness, cloud condensation nuclei, and upper-air thermal updrafts not fully captured in a single visible frame.
            2. **Domain Adaptation**: Sky camera models trained on one location (e.g. tropical maritime in Singapore / SWIMCAT) may degrade when deployed in temperate continental zones or urban settings with tall buildings.
            3. **Proxy Label Note**: In the absence of an on-site calibrated tipping-bucket rain gauge, reanalysis data (Open-Meteo) serves as a valuable research proxy, but must be recognized as reanalysis estimates rather than point-source physical ground truth.
            
            ### Recommended Operational Use
            - Use as an educational machine vision tool and supplemental localized sky monitoring aid.
            - Never use as an official emergency warning or flood mitigation mechanism.
            
            ### References & Credits
            - Liu et al., *Ground-based Remote Sensing Cloud Classification via Context Graph Attention Network*, TJNU Ground-based Cloud Dataset (2022).
            - Dev, S. et al., *SWIMCAT: Singapore Whole-sky IMaging CATegories Database*, ICIP (2015).
            - Open-Meteo Weather APIs (CC-BY 4.0).
            """
        )


if __name__ == "__main__":
    main()
