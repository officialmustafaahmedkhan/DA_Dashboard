    st.header("🧠 Deep Learning Suite — Multi-Layer Perceptron Neural Network (128 → 64 → 32)")

    dl1, dl2, dl3, dl4 = st.columns(4)
    with dl1:
        render_kpi_card("🧠 DL Test Accuracy", f"{bundle['dl']['accuracy']}%")
    with dl2:
        render_kpi_card("⚖️ DL Weighted F1", f"{bundle['dl']['f1']}%")
    with dl3:
        render_kpi_card("🏗️ Hidden Layers", "128 → 64 → 32")
    with dl4:
        render_kpi_card("⚡ Activation / Optimizer", "ReLU / Adam")

    st.markdown("---")
    dl_c1, dl_c2 = st.columns(2)

    with dl_c1:
        st.subheader("Neural Network Training · Loss & Validation Accuracy")
        epochs_range = list(range(1, len(bundle["dl"]["losses"]) + 1))
        fig_learn = go.Figure()
        fig_learn.add_trace(go.Scatter(
            x=epochs_range, y=bundle["dl"]["losses"], name="Training Loss",
            line=dict(color="#e63946", width=2.5)
        ))
        fig_learn.add_trace(go.Scatter(
            x=epochs_range, y=bundle["dl"]["val_accs"], name="Validation Accuracy (%)",
            yaxis="y2", line=dict(color="#0f4c81", width=2.5)
        ))
        fig_learn.update_layout(
            xaxis_title="Training Epoch",
            yaxis=dict(title="Cross-Entropy Loss"),
            yaxis2=dict(title="Accuracy (%)", overlaying="y", side="right"),
            height=410,
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_learn, width="stretch")

    with dl_c2:
        st.subheader("Deep Learning Multi-Class ROC Curves")
        y_te_bin = label_binarize(bundle["y_test"], classes=[0, 1, 2])
        y_probs_dl = bundle["dl"]["model"].predict_proba(bundle["X_test_p"])
        class_names = bundle["encoders"]["Category"].classes_

        fig_dl_roc = go.Figure()
        palette = ["#0f4c81", "#e63946", "#00a896"]
        for i, cls_label in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_probs_dl[:, i])
            fig_dl_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{cls_label} (AUC = {auc(fpr, tpr):.3f})",
                line=dict(color=palette[i], width=3)
            ))
        fig_dl_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="gray", dash="dash")))
        fig_dl_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=410)
        st.plotly_chart(fig_dl_roc, width="stretch")

    st.subheader("Deep Learning Confusion Matrix")
    dl_cm = confusion_matrix(bundle["y_test"], bundle["dl"]["model"].predict(bundle["X_test_p"]))
    fig_dl_cm = px.imshow(
        dl_cm, text_auto=True, x=class_names, y=class_names, color_continuous_scale="YlGnBu",
        labels=dict(x="Predicted Category", y="Actual Category", color="Samples")
    )
    fig_dl_cm.update_layout(height=380)
    st.plotly_chart(fig_dl_cm, width="stretch")

# ============================================================
# TAB 7: LIVE AI PREDICTOR
# ============================================================
with tab_pred:
    st.header("🤖 Live Multi-Paradigm AI Inference Studio")
    st.caption("Compare predictions between Deep Learning (MLP) and Supervised ML (Random Forest, KNN, Decision Tree, Logistic Regression).")

    i1, i2, i3, i4 = st.columns(4)
    in_loc = i1.selectbox("Location", bundle["encoders"]["Location"].classes_, key="p_loc")
    in_school = i2.selectbox("School Type", bundle["encoders"]["School Type"].classes_, key="p_sch")
    in_comp = i3.selectbox("Composition", bundle["encoders"]["Composition"].classes_, key="p_cmp")
    in_num = i4.number_input("Recorded Count (Number)", min_value=0.0, max_value=10000.0, value=320.0)

    all_inference_models = {"Deep Neural Network (MLP - DL)": bundle["dl"]["model"], **bundle["sup_models"]}

    i5, i6, i7 = st.columns([1, 1, 1.3])
    in_pop = i5.number_input("District Population", min_value=100000.0, max_value=5000000.0, value=1600000.0)
    in_rate = i6.number_input("Rate per 100k (No/100000)", min_value=0.0, max_value=1000.0, value=20.0)
    in_choice = i7.selectbox("Choose AI Model", list(all_inference_models.keys()))

    if st.button("🚀 Predict Category", type="primary"):
        row = pd.DataFrame([{
            "Location": bundle["encoders"]["Location"].transform([in_loc])[0],
            "School Type": bundle["encoders"]["School Type"].transform([in_school])[0],
            "Composition": bundle["encoders"]["Composition"].transform([in_comp])[0],
            "Number": in_num,
            "Population": in_pop,
            "No/100000": in_rate,
            "Log_Number": np.log1p(in_num)
        }])

        row_scaled = bundle["scaler"].transform(row)
        row_pca = bundle["pca_95"].transform(row_scaled)
        clf = all_inference_models[in_choice]

        pred_idx = clf.predict(row_pca)[0]
        probs = clf.predict_proba(row_pca)[0]
        pred_label = bundle["encoders"]["Category"].inverse_transform([pred_idx])[0]

        st.success(f"### Predicted Category ({in_choice}): **{pred_label}** (`{max(probs)*100:.2f}%` Confidence)")

        prob_df = pd.DataFrame({
            "Category": bundle["encoders"]["Category"].classes_,
            "Confidence (%)": (probs * 100).round(2)
        })
        fig_prob = px.bar(
            prob_df, x="Category", y="Confidence (%)", color="Category",
            text="Confidence (%)", color_discrete_sequence=["#0f4c81", "#00a896", "#e63946"]
        )
        fig_prob.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_prob, width="stretch")

# ------------------------------------------------------------
# 8. BRAND FOOTER
# ------------------------------------------------------------
st.markdown(
    '<div style="margin-top:24px;padding:14px 18px;border-radius:14px;background:linear-gradient(90deg,#0b1f46,#075985,#0d9488);color:white;text-align:center;font-weight:800;">'
    '💡 Smarter Insights. Better Education. A Brighter Sindh.'
    '</div>',
    unsafe_allow_html=True
)

