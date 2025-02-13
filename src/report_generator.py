import shap
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

class PDFReportGenerator:
    def __init__(self, model, data, shap_values, output_path='model_interpretability_report.pdf'):
        self.model = model
        self.data = data
        self.shap_values = shap_values
        self.output_path = output_path
        self.plots_dir = 'plots'
        self.reports_dir = 'reports'

        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)

    def generate_pdf_report(self):
        # Generate required plots
        self._generate_plots()

        # Initialize PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Model Interpretability Report', ln=True, align='C')

        # Model Info Section
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Model Information:', ln=True)
        pdf.set_font('Arial', '', 12)
        self._add_model_info(pdf)

        # Data Info Section
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Data Description:', ln=True)
        pdf.set_font('Arial', '', 12)
        self._add_data_info(pdf)

        # Add Plots
        self._add_plot_to_pdf(pdf, 'waterfall_plot.png', 'Waterfall Plot for Single Prediction',
                              "This plot illustrates how each feature contributes to a single prediction. "
                              "Positive SHAP values push the prediction higher, while negative values decrease it.")
        
        self._add_plot_to_pdf(pdf, 'bar_plot.png', 'Feature Importance Bar Plot',
                              "This bar plot ranks the features by their average SHAP values, indicating the overall "
                              "importance of each feature in making predictions.")
        
        self._add_plot_to_pdf(pdf, 'beeswarm_plot.png', 'Beeswarm Plot',
                              "The beeswarm plot is designed to display an information-dense summary of how the top" 
                              "features in a dataset impact the model output.")
        
        self._add_plot_to_pdf(pdf, 'heatmap_plot.png', 'Heatmap Plot',
                              "The heatmap plot function creates a plot with the instances on the x-axis, the model"
                              "inputs on the y-axis, and the SHAP values encoded on a color scale.")

        # Additional Section: Interpretation Summary
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Interpretation Summary:', ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10,
                       "The SHAP analysis provides both global and local interpretability for the model. "
                       "Globally, we understand which features are most influential across the entire dataset, "
                       "while locally, the waterfall plot allows us to dissect individual predictions.")

        # Save PDF
        pdf.output(f"{self.reports_dir}/{self.output_path}")
        print(f"PDF Report generated at {self.reports_dir}/{self.output_path}")

    def _add_model_info(self, pdf):
        # TensorFlow model summary or generic info for other models
        try:
            if 'keras' in str(type(self.model)).lower():
                import io
                stream = io.StringIO()
                self.model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
                model_summary = stream.getvalue()
                import re
                # Remove box-drawing characters (like ─, ├, ┤, └, etc.)
                model_summary = re.sub(r'[^\x20-\x7E]', '', model_summary)  
                pdf.multi_cell(0, 5, model_summary)
            else:
                pdf.multi_cell(0, 10, f"Model Type: {type(self.model).__name__}")
                pdf.multi_cell(0, 10, f"Model Parameters: {self.model.get_params()}")
        except Exception as e:
            pdf.multi_cell(0, 10, f"Could not retrieve detailed model information: {str(e)}")

    def _add_data_info(self, pdf):
        pdf.multi_cell(0, 10, f"Number of Samples: {self.data.shape[0]}")
        pdf.multi_cell(0, 10, f"Number of Features: {self.data.shape[1]}")
        pdf.multi_cell(0, 10, f"Feature Names: {', '.join(self.data.columns)}")
        pdf.multi_cell(0, 10, "This dataset was used to generate SHAP values, which help interpret the model predictions.")

    import matplotlib.pyplot as plt

    def _generate_plots(self):
        val = self.shap_values
        if len(self.shap_values.shape) > 2:
            val = self.shap_values[:, :, 0]

        # SHAP Waterfall Plot for First Prediction
        plt.figure(figsize=(6, 4))  # Create a new figure
        shap.plots.waterfall(val[0], show=False)  # Disable automatic display
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/waterfall_plot.png', bbox_inches='tight')  # Save the figure
        plt.close()  # Close the figure to prevent it from displaying

        # SHAP Feature Importance Bar Plot
        plt.figure(figsize=(6, 4))  # Create a new figure
        shap.plots.bar(val, show=False)  # Disable automatic display
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/bar_plot.png', bbox_inches='tight')  # Save the figure
        plt.close()  # Close the figure to prevent it from displaying

        # SHAP Feature Importance Bar Plot
        plt.figure(figsize=(6, 4))  # Create a new figure
        shap.plots.heatmap(val, show=False)  # Disable automatic display
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/heatmap_plot.png', bbox_inches='tight')  # Save the figure
        plt.close()  # Close the figure to prevent it from displaying

        # SHAP Feature Importance Bar Plot
        plt.figure(figsize=(6, 4))  # Create a new figure
        shap.plots.beeswarm(val, show = False)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/beeswarm_plot.png', bbox_inches='tight')  # Save the figure
        plt.close()  # Close the figure to prevent it from displaying





    def _add_plot_to_pdf(self, pdf, plot_filename, title, description):
        plot_path = f'{self.plots_dir}/{plot_filename}'
        if os.path.exists(plot_path):
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, title, ln=True)
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 10, description)
            pdf.image(plot_path, w=pdf.w - 40)

