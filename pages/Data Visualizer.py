import pandas as pd
import openai
import streamlit as st
import os
import dotenv
from dotenv import load_dotenv
load_dotenv()
# from nbconvert import HTMLExporter
# import nbformat


#import streamlit_nested_layout
import warnings
warnings.filterwarnings("ignore")
st.title("ChatGPT Inventory ChatBot Visualizer")
st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_page_config(page_icon="chat2vis.png",layout="wide",page_title="Chat2VIS")

available_models = {"ChatGPT-4": "gpt-4"}

openai_key = os.environ.get("OPENAI_API_KEY")         # API Key
hf_key = os.environ.get("HUGGINGFACEAPIKEY") 

# openai_key = ""
# hf_key = ""

chosen_dataset = pd.read_csv("C:/Users/roopa/Downloads/streamlit2/purchase_order.csv")

from langchain import HuggingFaceHub, LLMChain,PromptTemplate

def run_request(question_to_ask, model_type, key, alt_key):
    if model_type == "gpt-4" or model_type == "gpt-3.5-turbo" :
        # Run OpenAI ChatCompletion API
        task = "Generate Python Code Script."
        if model_type == "gpt-4":
            # Ensure GPT-4 does not include additional comments
            task = task + " The script should only include code, no comments."
        openai.api_key = key
        response = openai.ChatCompletion.create(model=model_type,
            messages=[{"role":"system","content":task},{"role":"user","content":question_to_ask}])
        llm_response = response["choices"][0]["message"]["content"]
    elif model_type == "text-davinci-003" or model_type == "gpt-3.5-turbo-instruct":
        # Run OpenAI Completion API
        openai.api_key = key
        response = openai.Completion.create(engine=model_type,prompt=question_to_ask,temperature=0,max_tokens=500,
                    top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,stop=["plt.show()"])
        llm_response = response["choices"][0]["text"] 
    else:
        # Hugging Face model
        llm = HuggingFaceHub(huggingfacehub_api_token = alt_key, repo_id="codellama/" + model_type, model_kwargs={"temperature":0.1, "max_new_tokens":500})
        llm_prompt = PromptTemplate.from_template(question_to_ask)
        llm_chain = LLMChain(llm=llm,prompt=llm_prompt)
        llm_response = llm_chain.predict()
    # rejig the response
    llm_response = format_response(llm_response)
    return llm_response

def format_response( res):
    # Remove the load_csv from the answer if it exists
    csv_line = res.find("read_csv")
    if csv_line > 0:
        return_before_csv_line = res[0:csv_line].rfind("\n")
        if return_before_csv_line == -1:
            # The read_csv line is the first line so there is nothing to need before it
            res_before = ""
        else:
            res_before = res[0:return_before_csv_line]
        res_after = res[csv_line:]
        return_after_csv_line = res_after.find("\n")
        if return_after_csv_line == -1:
            # The read_csv is the last line
            res_after = ""
        else:
            res_after = res_after[return_after_csv_line:]
        res = res_before + res_after
    return res

def format_question(primer_desc,primer_code , question, model_type):
    # Fill in the model_specific_instructions variable
    instructions = ""
    if model_type == "Code Llama":
        # Code llama tends to misuse the "c" argument when creating scatter plots
        instructions = "\nDo not use the 'c' argument in the plot function, use 'color' instead and only pass color names like 'green', 'red', 'blue'."
    primer_desc = primer_desc.format(instructions)  
    # Put the question at the end of the description primer within quotes, then add on the code primer.
    return  '"""\n' + primer_desc + question + '\n"""\n' + primer_code

def get_primer(df_dataset,df_name):
    # Primer function to take a dataframe and its name
    # and the name of the columns
    # and any columns with less than 20 unique values it adds the values to the primer
    # and horizontal grid lines and labeling
    primer_desc = "Use a dataframe called df from data_file.csv with columns '" \
        + "','".join(str(x) for x in df_dataset.columns) + "'. "
    for i in df_dataset.columns:
        if len(df_dataset[i].drop_duplicates()) < 20 and df_dataset.dtypes[i]=="O":
            primer_desc = primer_desc + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df_dataset[i].drop_duplicates()) + "'. "
        elif df_dataset.dtypes[i]=="int64" or df_dataset.dtypes[i]=="float64":
            primer_desc = primer_desc + "\nThe column '" + i + "' is type " + str(df_dataset.dtypes[i]) + " and contains numeric values. "   
    primer_desc = primer_desc + "\nLabel the x and y axes appropriately."
    primer_desc = primer_desc + "\nAdd a title. Set the fig suptitle as empty."
    primer_desc = primer_desc + "{}" # Space for additional instructions if needed
    primer_desc = primer_desc + "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    pimer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    pimer_code = pimer_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    pimer_code = pimer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    pimer_code = pimer_code + "df=" + df_name + ".copy()\n"
    return primer_desc,pimer_code

 # Text area for query
question = st.text_area(":eyes: What would you like to visualise?",height=10)
go_btn = st.button("Go...")

# Make a list of the models which have been selected
selected_models = ["ChatGPT-4"]
model_count = len(selected_models)

# Execute chatbot query
if go_btn and model_count > 0:
    api_keys_entered = True
    # Check API keys are entered.
    if  "ChatGPT-4" in selected_models:
        if not openai_key.startswith('sk-'):
            st.error("Please enter a valid OpenAI API key.")
            api_keys_entered = False
    if "Code Llama" in selected_models:
        if not hf_key.startswith('hf_'):
            st.error("Please enter a valid HuggingFace API key.")
            api_keys_entered = False
    if api_keys_entered:
        # Place for plots depending on how many models
        plots = st.columns(model_count)
        # Get the primer for this dataset
        primer1,primer2 = get_primer(chosen_dataset,'chosen_dataset') 
        # Create model, run the request and print the results
        for plot_num, model_type in enumerate(selected_models):
            with plots[plot_num]:
                st.subheader(model_type)
                try:
                    # Format the question 
                    question_to_ask = format_question(primer1, primer2, question, model_type)   
                    # Run the question
                    answer=""
                    answer = run_request(question_to_ask, available_models[model_type], key=openai_key,alt_key=hf_key)
                    # the answer is the completed Python script so add to the beginning of the script to it.
                    answer = primer2 + answer
                    print("Model: " + model_type)
                    print(answer)
                    plot_area = st.empty()
                    plot_area.pyplot(exec(answer))           
                except Exception as e:
                    st.write("Sorry! I don't have sufficent info")
                    
