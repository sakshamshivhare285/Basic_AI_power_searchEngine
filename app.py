import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import  load_dotenv

# side bar setting
st.sidebar.title("settings")
api_key = st.sidebar.text_input('Enter your groq  api key:',type='password')


# used inbuilt tool
api_wraper_wiki = WikipediaAPIWrapper(top_k_results=5,doc_content_chars_max=500)
api_wraper_arxi = ArxivAPIWrapper(top_k_results=5,doc_content_chars_max=500)

wiki_tool = WikipediaQueryRun(api_wrapper=api_wraper_wiki)
arxi_tool = ArxivQueryRun(api_wrapper=api_wraper_arxi)

# search 
search = DuckDuckGoSearchRun(name="search")


st.title("Langchain with search")


if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role':"assistant",
         "content":"hi, I am AI powered search Engine"}
    ]

for mgs in st.session_state.messages:
    st.chat_message(mgs['role']).write(mgs['content'])

if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({'role':'user',"content":prompt})
    st.chat_message('user').write(prompt)

    llm = ChatGroq(api_key=api_key,model="Llama3-8b-8192",streaming=True)
    tools = [search,arxi_tool,wiki_tool]

    # creating search agent
    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handling_parsing_error=True)

    with st.chat_message('assistant'):
        st_callbacks = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        response = search_agent.run(st.session_state.messages,callbacks=[st_callbacks])
        st.session_state.messages.append({'role':'assistant',"content": response})
        st.write(response)