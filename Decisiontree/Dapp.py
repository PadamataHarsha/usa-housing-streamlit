from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).resolve().parent / "kyphosis.csv"
TARGET_COLUMN = "Kyphosis"
FEATURE_COLUMNS = ["Age", "Number", "Start"]
CLASS_LABELS = ["absent", "present"]

st.set_page_config(
    page_title="Kyphosis Classifier",
    page_icon="🩺",
    layout="wide",
)


@st.cache_data
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the local CSV once and cache its contents."""
    return pd.read_csv(csv_path)


@st.cache_resource
def train_models(data: pd.DataFrame) -> dict:
    """Train both classifiers using the same reproducible train/test split."""
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=42,
        stratify=target,
    )

    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)

    return {
        "decision_tree": decision_tree,
        "random_forest": random_forest,
        "X_test": X_test,
        "y_test": y_test,
    }


def show_pairplot(data: pd.DataFrame) -> None:
    """Display pairwise feature relationships grouped by the target."""
    pairplot = sns.pairplot(data, hue=TARGET_COLUMN, palette="Set1")
    st.pyplot(pairplot.fig, use_container_width=True)
    plt.close(pairplot.fig)


def main() -> None:
    st.title("Kyphosis Classification")
    st.caption("Explore the dataset, compare classifiers, and generate a prediction.")

    # Fail clearly and stop before attempting to load or train on a missing file.
    if not DATA_PATH.is_file():
        st.error(f"Dataset file not found: {DATA_PATH.name}. Place it beside Dapp.py.")
        st.stop()

    try:
        data = load_dataset(str(DATA_PATH))
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not read the dataset: {error}")
        st.stop()

    required_columns = {TARGET_COLUMN, *FEATURE_COLUMNS}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        st.error(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")
        st.stop()

    model_bundle = train_models(data)

    with st.sidebar:
        st.header("Prediction Inputs")
        selected_model_name = st.selectbox(
            "Model", ["Decision Tree", "Random Forest"]
        )
        age = st.number_input("Age", min_value=0, max_value=200, value=25)
        number = st.number_input("Number", min_value=1, max_value=20, value=3)
        start = st.number_input("Start", min_value=1, max_value=25, value=5)
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    st.header("Dataset Overview")
    overview_cols = st.columns(3)
    overview_cols[0].metric("Rows", f"{data.shape[0]:,}")
    overview_cols[1].metric("Columns", f"{data.shape[1]}")
    overview_cols[2].metric("Present cases", f"{(data[TARGET_COLUMN] == 'present').sum()}")
    st.dataframe(data.head(), use_container_width=True, hide_index=True)
    st.caption(f"Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")

    st.header("EDA Visualizations")
    with st.expander("Pairplot", expanded=True):
        show_pairplot(data)

    with st.expander("Correlation Heatmap", expanded=True):
        heatmap_fig, heatmap_ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            data[FEATURE_COLUMNS].corr(),
            annot=True,
            cmap="crest",
            fmt=".2f",
            ax=heatmap_ax,
        )
        heatmap_ax.set_title("Feature Correlations")
        st.pyplot(heatmap_fig, use_container_width=True)
        plt.close(heatmap_fig)

    st.header("Model Training")
    st.write(
        "Both classifiers are trained on a stratified 70/30 train/test split. "
        "The random forest uses 100 trees."
    )
    st.success("Decision Tree and Random Forest are trained and ready.")

    selected_model = (
        model_bundle["decision_tree"]
        if selected_model_name == "Decision Tree"
        else model_bundle["random_forest"]
    )

    st.header("Prediction")
    if predict_clicked:
        input_data = pd.DataFrame(
            [[age, number, start]],
            columns=FEATURE_COLUMNS,
        )
        prediction = selected_model.predict(input_data)[0]
        result_label = "Present" if prediction == "present" else "Absent"
        if prediction == "present":
            st.error(f"Prediction: {result_label}")
        else:
            st.success(f"Prediction: {result_label}")
    else:
        st.info("Enter patient values in the sidebar and select Predict.")

    st.header("Model Evaluation")
    test_predictions = selected_model.predict(model_bundle["X_test"])
    accuracy = accuracy_score(model_bundle["y_test"], test_predictions)
    report = classification_report(
        model_bundle["y_test"],
        test_predictions,
        labels=CLASS_LABELS,
        target_names=["Absent", "Present"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        model_bundle["y_test"], test_predictions, labels=CLASS_LABELS
    )

    evaluation_cols = st.columns(3)
    evaluation_cols[0].metric("Accuracy", f"{accuracy:.1%}")
    evaluation_cols[1].metric("Test samples", f"{len(model_bundle['y_test'])}")
    evaluation_cols[2].metric("Selected model", selected_model_name)

    report_data = pd.DataFrame(report).transpose().round(3)
    st.subheader("Classification Report")
    st.dataframe(report_data, use_container_width=True)

    matrix_col, tree_col = st.columns([1, 2])
    with matrix_col:
        st.subheader("Confusion Matrix")
        matrix_fig, matrix_ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="crest",
            xticklabels=["Absent", "Present"],
            yticklabels=["Absent", "Present"],
            ax=matrix_ax,
        )
        matrix_ax.set_xlabel("Predicted")
        matrix_ax.set_ylabel("Actual")
        st.pyplot(matrix_fig, use_container_width=True)
        plt.close(matrix_fig)

    with tree_col:
        st.subheader("Decision Tree Visualization")
        tree_fig, tree_ax = plt.subplots(figsize=(15, 7))
        tree.plot_tree(
            model_bundle["decision_tree"],
            feature_names=FEATURE_COLUMNS,
            class_names=["Absent", "Present"],
            filled=True,
            rounded=True,
            ax=tree_ax,
        )
        tree_ax.set_title("Trained Decision Tree")
        st.pyplot(tree_fig, use_container_width=True)
        plt.close(tree_fig)


if __name__ == "__main__":
    main()