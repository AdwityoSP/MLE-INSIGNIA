import gradio as gr
import pandas as pd
import joblib

try:
    model = joblib.load('lasso_lars_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("The model file 'lasso_lars_model.pkl' was not found. Please ensure it exists in the working directory.")

def predict_price(length, length_unit, width, width_unit, is_carport):
    if length <= 0 or width <= 0:
        return "Error: Length and Width must be positive numbers."

    if length_unit.lower() == 'centimeter':
        length_m = length / 100.0
    elif length_unit.lower() == 'meter':
        length_m = length
    else:
        return "Error: Unknown length unit. Use 'meter' or 'centimeter'."

    if width_unit.lower() == 'centimeter':
        width_m = width / 100.0
    elif width_unit.lower() == 'meter':
        width_m = width
    else:
        return "Error: Unknown width unit. Use 'meter' or 'centimeter'."

    input_df = pd.DataFrame({
        'length_m': [length_m],
        'width_m': [width_m],
        'isCarport': [is_carport]
    })

    try:
        predicted_price = model.predict(input_df)[0]
    except Exception as e:
        return f"Prediction Error: {str(e)}"

    return f"Predicted Price: {predicted_price:.2f} Billion"

iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Length (numeric value)", value=20, precision=2),
        gr.Dropdown(choices=["meter", "centimeter"], label="Length Unit", value="meter"),
        gr.Number(label="Width (numeric value)", value=10, precision=2),
        gr.Dropdown(choices=["meter", "centimeter"], label="Width Unit", value="meter"),
        gr.Radio(choices=["0", "1"], label="Is Carport?", value="1") 
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="🏠 House Price Prediction",
    description="Enter the details of the house to predict its price in billions.",
    examples=[
        [20, "meter", 10, "meter", "1"],
        [40, "meter", 20, "meter", "0"],
        [3000, "centimeter", 2000, "centimeter", "1"],
        [1000, "centimeter", 3000, "centimeter", "0"]
    ],
    theme="default",
    allow_flagging="never"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
