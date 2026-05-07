import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="Movie Info Extractor",
    page_icon="🎬",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #6366f1;
    background-color: #1e293b;
    color: white;
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    background-color: #6366f1;
    color: white;
    font-size: 18px;
    padding: 10px;
    border: none;
}
.stButton>button:hover {
    background-color: #4f46e5;
}
.result-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 16px;
    margin-top: 20px;
    border: 1px solid #334155;
}
.result-title {
    color: #818cf8;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1 style='text-align:center; color:#818cf8;'>🎬 Movie Information Extractor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:lightgray;'>Extract movie details instantly using LangChain + OpenAI</p>",
    unsafe_allow_html=True
)

# User Input
para = st.text_area(
    "Enter Movie Paragraph",
    height=220,
    placeholder="Paste your movie paragraph here..."
)

# Pydantic Schema
class Movie(BaseModel):
    name: str
    director: str
    cast_members: List[str]
    genre: List[str]
    summary: Optional[str]

# Parser
parser = PydanticOutputParser(pydantic_object=Movie)

# Model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Prompt Template
movie_info_prompt = PromptTemplate(
    input_variables=["para"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
    template="""
You are an intelligent movie information extraction assistant.

Read the given movie paragraph carefully and extract:

- Movie Name
- Director
- Cast Members
- Genre
- Small Summary

Instructions:
- If any information is missing, write "Not Mentioned".
- Keep the summary short and clear.
- Extract only relevant information from the paragraph.

Movie Paragraph:
{para}

{format_instructions}
"""
)

# Button
if st.button("Extract Movie Information"):

    if para.strip() == "":
        st.warning("Please enter a movie paragraph.")
    else:

        with st.spinner("Extracting information..."):

            formatted_prompt = movie_info_prompt.format(para=para)

            response = model.invoke(formatted_prompt)

            parsed_output = parser.parse(response.content)

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        st.markdown(
            f"<p class='result-title'>🎥 Movie Name</p>",
            unsafe_allow_html=True
        )
        st.write(parsed_output.name)

        st.markdown(
            f"<p class='result-title'>🎬 Director</p>",
            unsafe_allow_html=True
        )
        st.write(parsed_output.director)

        st.markdown(
            f"<p class='result-title'>⭐ Cast Members</p>",
            unsafe_allow_html=True
        )
        st.write(", ".join(parsed_output.cast_members))

        st.markdown(
            f"<p class='result-title'>🎭 Genre</p>",
            unsafe_allow_html=True
        )
        st.write(", ".join(parsed_output.genre))

        st.markdown(
            f"<p class='result-title'>📝 Summary</p>",
            unsafe_allow_html=True
        )
        st.write(parsed_output.summary)

        st.markdown("</div>", unsafe_allow_html=True)