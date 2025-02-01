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
            padding: 10px;
            border-radius: 5px;
            border-bottom: 1px solid #444444;
        }
        .row-odd {
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            border-bottom: 1px solid #444444;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Transaction Labeller")

    # Initialize session state for categories if not exists
    if 'categories' not in st.session_state:
        st.session_state.categories = {}

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
            # Add column headers
            header_col1, header_col2, header_col3, header_col4, header_col5, header_col6 = st.columns([1, 1, 1, 2, 2, 2])
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

            # Display each transaction with its predictions
            for idx, (transaction, pred_categories) in enumerate(zip(df['Name'], predictions)):
                row_class = "row-even" if idx % 2 == 0 else "row-odd"
                st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)
                col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 2, 2, 2])
                
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
                    # Display predicted categories as buttons in a row
                    button_cols = st.columns(len(pred_categories))
                    button_clicked = False
                    clicked_category = None
                    for button_col, category in zip(button_cols, pred_categories):
                        with button_col:
                            if st.button(category, key=f"pred_{idx}_{category}"):
                                button_clicked = True
                                clicked_category = category
                
                with col6:
                    # Dropdown for manual category selection
                    all_categories = sorted(model.classes_)
                    # Get category from session state or dataframe
                    if idx not in st.session_state.categories:
                        current_category = df.at[idx, 'Category']
                        if not current_category:
                            current_category = pred_categories[0] if pred_categories else all_categories[0]
                        st.session_state.categories[idx] = current_category
                    
                    # Update category if button was clicked
                    if button_clicked:
                        st.session_state.categories[idx] = clicked_category
                    selected_category = st.selectbox(
                        "Select category",
                        all_categories,
                        index=all_categories.index(st.session_state.categories[idx]) if st.session_state.categories[idx] in all_categories else 0,
                        key=f"dropdown_{idx}"
                    )
                    st.session_state.categories[idx] = selected_category
                    df.at[idx, 'Category'] = selected_category
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
