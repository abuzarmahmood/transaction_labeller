import streamlit as st
import pandas as pd
from pred_transactions import return_model_and_vectorizer, predict_categories

def main():
    st.set_page_config(page_title="Transaction Labeller", layout="wide")
    
    # Add CSS for alternating row colors
    st.markdown("""
        <style>
        .row-even {
            background-color: #f0f2f6;
            border-radius: 5px;
            border-bottom: 1px solid #444444;
        }
        .row-odd {
            padding: 10px;
            border-radius: 5px;
            border-bottom: 1px solid #444444;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Transaction Labeller")

    # Initialize session state for categories and flags if not exists
    if 'categories' not in st.session_state:
        st.session_state.categories = {}
    if 'flags' not in st.session_state:
        st.session_state.flags = {}

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
        
        # Add Category and Flag columns if they don't exist
        if 'Category' not in df.columns:
            df['Category'] = ''
        if 'Flag' not in df.columns:
            df['Flag'] = False

        # Get predictions and probabilities for all transactions
        predictions, probabilities = predict_categories(model, vectorizer, df['Name'].values, n=5)
        
        # Initialize categories with top predictions
        for idx, (pred_categories, _) in enumerate(zip(predictions, probabilities)):
            if not df.at[idx, 'Category'] and pred_categories:
                df.at[idx, 'Category'] = pred_categories[0]

        # Create a container for the transactions
        transactions_container = st.container()

        col_format = [0.5, 0.5, 1, 0.5, 2.5, 1, 0.5]
        
        with transactions_container:
            # Add column headers
            header_col1, header_col2, header_col3, header_col4, header_col5, header_col6, header_col7 = st.columns(col_format)
            with header_col1:
                if 'Date' in df.columns:
                    st.markdown("**Date**")
            with header_col2:
                if 'Amount' in df.columns:
                    st.markdown("**Amount**")
            with header_col3:
                if 'Account' in df.columns:
                    st.markdown("**Account**")
            with header_col4:
                st.markdown("**Transaction**")
            with header_col5:
                st.markdown("**Suggestions**")
            with header_col6:
                st.markdown("**Category**")
            with header_col7:
                st.markdown("**Flag**")

            # Display each transaction with its predictions
            for idx, (transaction, pred_categories) in enumerate(zip(df['Name'], predictions)):
                row_class = "row-even" if idx % 2 == 0 else "row-odd"
                st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)
                col1, col2, col3, col4, col5, col6, col7 = st.columns(col_format)
                
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
                    # Display Account if present
                    if 'Account' in df.columns:
                        st.text(str(df.at[idx, 'Account']))
                    else:
                        st.text("")

                with col4:
                    st.text(transaction)
                
                with col5:
                    # Display predicted categories as buttons in a row with probabilities
                    button_cols = st.columns(len(pred_categories))
                    button_clicked = False
                    clicked_category = None
                    current_category = st.session_state.categories.get(idx, '')
                    
                    # Only show buttons for predictions with >= 5% probability
                    filtered_predictions = [(cat, prob) for cat, prob in zip(pred_categories, probabilities[idx]) if prob >= 0.05]
                    if filtered_predictions:
                        button_cols = st.columns(len(filtered_predictions))
                    
                    for button_col, (category, prob) in zip(button_cols, filtered_predictions):
                        with button_col:
                            is_selected = current_category == category
                            if st.button(f"{category}\n{prob:.0%}", key=f"pred_{idx}_{category}"):
                                button_clicked = True
                                clicked_category = category
                
                with col6:
                    # Dropdown for manual category selection
                    all_categories = sorted(model.classes_)
                    # Get category from session state or dataframe
                    if idx not in st.session_state.categories:
                        current_category = df.at[idx, 'Category']
                        if not current_category:
                            if pred_categories:
                                current_category = pred_categories[0]
                            else:
                                current_category = None
                            # current_category = pred_categories[0] if pred_categories else all_categories[0]
                        st.session_state.categories[idx] = current_category
                    
                    # Update category if button was clicked
                    if button_clicked:
                        st.session_state.categories[idx] = clicked_category
                        current_category = st.session_state.categories[idx]
                    else:
                        current_category = pred_categories[0]
                    if current_category not in all_categories: # If category is already selected, show it
                        list_idx = 0
                    else:
                        list_idx = all_categories.index(current_category)
                    selected_category = st.selectbox(
                        "Select category",
                        all_categories,
                        index=list_idx,
                        key=f"dropdown_{idx}"
                    )
                    st.session_state.categories[idx] = selected_category
                    df.at[idx, 'Category'] = selected_category
                
                with col7:
                    # Checkbox for flagging transactions
                    if idx not in st.session_state.flags:
                        st.session_state.flags[idx] = False
                    
                    is_flagged = st.checkbox(
                        "",
                        value=st.session_state.flags[idx],
                        key=f"flag_{idx}"
                    )
                    st.session_state.flags[idx] = is_flagged
                    df.at[idx, 'Flag'] = is_flagged
                    
                st.markdown('</div>', unsafe_allow_html=True)
        
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
