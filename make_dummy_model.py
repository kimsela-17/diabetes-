
import joblib
import random

class DummyModel:
    def predict(self, X):
        # X is a list of feature lists; return deterministic-ish result based on glucose for demo
        out = []
        for features in X:
            glucose = features[0]
            if glucose >= 200:
                out.append(2)  # Diabetes
            elif glucose >= 140:
                out.append(1)  # Pre-diabetes
            else:
                # Randomly choose no risk vs pre-diabetes for mid range to make demo varied
                out.append(0)
        return out

if __name__ == "__main__":
    model = DummyModel()
    joblib.dump(model, "diabetes_model.pkl")
    print("Dummy model saved to diabetes_model.pkl")
