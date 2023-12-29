import os
import streamlit as st
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.memory import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
import dotenv
from dotenv import load_dotenv
load_dotenv()
# from nbconvert import HTMLExporter
# import nbformat

# OPENAI_API_KEY = os.environ.get("OPENAIAPIKEY")         # API Key
# SERPAPI_API_KEY = os.environ.get("SERPAPI") 

# os.environ['OPENAI_API_KEY'] = "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"
# os.environ['SERPAPI_API_KEY'] = "74963d51b78c062e4bcadf618e77acbe264485d2763b9478161ca0823169d818"

st.set_page_config(
    page_title="Inventory Assistant",
    page_icon = "👋"
)
st.title("ChatGPT Inventory ChatBot")
st.sidebar.success("Select a page above")

from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "I have 9 tables in my database namely purchase_order,purchase_order_item, purchase_order_item_details, asn, stores, category, products, product_details, suppliers purchase_order have the following fields:- po_number, status, asn_id, supplier_id, expected_qty, received_qty, creation_date, total_cost, no_of_items."+
"asn have the fileds namely:- asn_number, date, quantity, status, no_of_container, container_details." +
"category have fields category_id, category_name. products have fields:- item_number,item_name,category_id."+
"stores table have fileds namingly store_id,store_address,store_name,store_stock."+
"suppliers have fields supplier_id,supplier_name."+
"purchase_order_item have fileds item_number,category, expected_qty,item_name,po_number,received_qty."+
"purchase_order_item_details have fields id, colour, image_data, price, size, stock, store, item_number."+
"product_details have fields id, colour, image_data, price, size, stock, store_id, item_number."+
  
" purchase order, purchaseorder, PURCHASE ORDER, PURCHASEORDER, Purchase Order and PurchaseOrder needs to be considered as one and the same thing"+

" purchaseorder number, order number can be considered same and is defined as po_number"+

"po_number can also be treated as po_id and po_no. & should be treated as and. cost is same as total_cost."+

"Handbags & handbags should be considered as same. Footwears & footwears are same words. Watches & watches are same. Totebags should be treated same as totebag."+
"Sportswear & sportswear are same words.Cosmetics is same as cosmetics.Luggage is same as luggage.Jewellery & jewellery are same words."+
"Accessories & accessories should be treated same.Bagpacks should be treated same as bagpacks."

)

from langchain.prompts import ChatPromptTemplate
template = """I have 9 tables in my database namely purchase_order,purchase_order_item, purchase_order_item_details, asn, stores, category, products, product_details, suppliers purchase_order have the following fields:- po_number, status, asn_id, supplier_id, expected_qty, received_qty, creation_date, total_cost, no_of_items.
asn have the fileds namely:- asn_number, date, quantity, status, no_of_container, container_details.
category have fields category_id, category_name. products have fields:- item_number,item_name,category_id.
stores table have fileds namingly store_id,store_address,store_name,store_stock.
suppliers have fields supplier_id,supplier_name.
purchase_order_item have fileds item_number,category, expected_qty,item_name,po_number,received_qty.
purchase_order_item_details have fields id, colour, image_data, price, size, stock, store, item_number.
product_details have fields id, colour, image_data, price, size, stock, store_id, item_number.
  
purchase order, purchaseorder, PURCHASE ORDER, PURCHASEORDER, Purchase Order and PurchaseOrder needs to be considered as one and the same thing

purchaseorder number, order number can be considered same and is defined as po_number

po_number can also be treated as po_id and po_no. & should be treated as and. cost is same as total_cost.

Handbags & handbags should be considered as same. Footwears & footwears are same words. Watches & watches are same. Totebags should be treated same as totebag.
Sportswear & sportswear are same words.Cosmetics is same as cosmetics.Luggage is same as luggage.Jewellery & jewellery are same words.
Accessories & accessories should be treated same.Bagpacks should be treated same as bagpacks."""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
db = SQLDatabase.from_uri("sqlite:///inventorymanagementdb (5).db")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True,top_k=100)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    Tool(
        name="Inventory-Database",
        func=db_chain.run,
        description="useful for when you need to answer questions about Inventory Database. Input should be in the form of a question containing full context",
    ),
]


chat_history_size = 3   # store 3 recent chat messages
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=msgs, k=chat_history_size, return_messages=True)
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.OPENAI_FUNCTIONS, 
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)
if "messages" not in st.session_state.keys():
    st.session_state.messages =[
        {"role" :"assistant","content":"Hello I am KPMG Bot, How can I assist you?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role":"user","content":user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
               
               try:  
                ai_response = agent.run(user_prompt) 
                st.write(ai_response)
                new_ai_message = {"role":"assistant","content":ai_response}
                st.session_state.messages.append(new_ai_message)
               except:
                ai_response = "Please provide relevant information!"
                st.write(ai_response)
                new_ai_message = {"role":"assistant","content":ai_response}
                st.session_state.messages.append(new_ai_message)

                   
                   
                
    

    
    
    