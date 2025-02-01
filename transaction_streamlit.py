import streamlit as st
import pandas as pd
from pred_transactions import return_model_and_vectorizer, predict_categories

def main():
    st.set_page_config(page_title="Transaction Labeller", layout="wide")
    st.title("Transaction Labeller")

    # Load model and vectorizer
    @st.cache_resource
    def load_model():
        return return_model_and_vectorizer()
    
    model, vectorizer = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        # Load the data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Add Category column if it doesn't exist
        if 'Category' not in df.columns:
            df['Category'] = ''

        # Get predictions for all transactions
        predictions = predict_categories(model, vectorizer, df['Name'].values, n=5)
        
        # Create a container for the transactions
        transactions_container = st.container()
        
        with transactions_container:
            # Display each transaction with its predictions
            for idx, (transaction, pred_categories) in enumerate(zip(df['Name'], predictions)):
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 2, 2])
                
                with col1:
                    # Display Date if present
                    if 'Date' in df.columns:
                        st.text(str(df.at[idx, 'Date']))
                    else:
                        st.text("")

                with col2:
                    # Display Amount if present
                    if 'Amount' in df.columns:
                        st.text(f"${df.at[idx, 'Amount']:.2f}")
                    else:
                        st.text("")

                with col3:
                    st.text(transaction)
                
                with col2:
                    # Display predicted categories as buttons in a row
                    button_cols = st.columns(len(pred_categories))
                    for button_col, category in zip(button_cols, pred_categories):
                        with button_col:
                            if st.button(category, key=f"pred_{idx}_{category}"):
                                df.at[idx, 'Category'] = category
                                st.experimental_rerun()
                
                with col3:
                    # Dropdown for manual category selection
                    all_categories = sorted(model.classes_)
                    current_category = df.at[idx, 'Category']
                    # If no category is set, use the most likely predicted category
                    if not current_category and pred_categories:
                        current_category = pred_categories[0]
                    selected_category = st.selectbox(
                        "Select category",
                        all_categories,
                        index=all_categories.index(current_category) if current_category in all_categories else 0,
                        key=f"dropdown_{idx}"
                    )
                    df.at[idx, 'Category'] = selected_category
        
        # Save button
        if st.button("Save Results"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="labeled_transactions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
