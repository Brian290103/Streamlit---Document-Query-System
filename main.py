import streamlit as st

st.title("Ask Point Project -  WebScrapper")

url = st.text_input("Enter Website URL")

if st.button("Scrape Website"):
    if url:
        st.write("Scraping the website...")