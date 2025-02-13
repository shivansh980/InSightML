import shap

# Context class for strategy selection
class ShapExplainerContext:
    def __init__(self, model, data):
        self.model = model
        self.data = data
    

    def explain_model(self):
        explainer = shap.Explainer(self.model, self.data)
        shap_values = explainer(self.data)
        return shap_values
