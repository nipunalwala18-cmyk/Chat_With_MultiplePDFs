import streamlit as st


def main():
    st.set_page_config(page_title="Chat with multiple PD ")
    print("Hello World!")

    st.header("Chat with Multiple PDFs: books:")
    st.text_input("Ask a question about your documents: ")

    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload your PDFS here an click on Process")
        st.button("Process")

if __name__ == "__main__":
    main()