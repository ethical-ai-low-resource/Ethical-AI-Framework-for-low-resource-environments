import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Saved artifacts ko safe tarike se load karein
    # Simple logic: Files ki presence check karke model aur vectorizer memory mein lana
    try:
        print("Loading model and vectorizer...")
        model = joblib.load('urdu_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
    except FileNotFoundError as e:
        print(f"Error: Missing required file -> {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        sys.exit(1)

    # 2. Features (Words) aur Model Coefficients extract karein
    # Simple logic: Model ne kis word ko kitna weightage diya hai woh calculate karna
    try:
        feature_names = vectorizer.get_feature_names_out()
        
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
        else:
            raise AttributeError("Loaded model does not contain linear coefficients ('coef_').")

    except Exception as e:
        print(f"Error extracting features: {e}")
        sys.exit(1)

    # 3. Top 15 positive aur negative impact wale words filter karein
    # Simple logic: Top positive (green) aur top negative (red) words chun-na
    top_n = 15
    top_negative_idx = np.argsort(coefficients)[:top_n]
    top_positive_idx = np.argsort(coefficients)[-top_n:]

    top_indices = np.hstack([top_negative_idx, top_positive_idx])
    top_features = feature_names[top_indices]
    top_coefficients = coefficients[top_indices]

    # 4. Professional Visual Chart
    # Simple logic: Horizontal bar chart banana taake text clear aur readable rahe
    plt.figure(figsize=(12, 7))
    
    # Negative impact = Soft Red (#d9534f), Positive impact = Soft Green (#5cb85c)
    colors = ['#d9534f' if c < 0 else '#5cb85c' for c in top_coefficients]
    
    plt.barh(range(len(top_coefficients)), top_coefficients, color=colors, edgecolor='black', linewidth=0.5)
    plt.yticks(range(len(top_coefficients)), top_features, fontsize=10)
    
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('Coefficient Value (Impact on Model Decision)', fontsize=11, fontweight='bold')
    plt.title(f'Top {top_n} Positive and Negative Word Features (Feature Importance)', fontsize=13, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()

    # 5. Image Ko High Quality Mein Save Karein
    output_filename = 'shap_feature_importance.png'
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot successfully generated and saved as '{output_filename}'")
    except Exception as e:
        print(f"Failed to save plot image: {e}")

if __name__ == "__main__":
    main()