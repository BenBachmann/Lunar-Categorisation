## the pipeline is described here:

## for a dataframe (assume we already have website summaries):
# filter by crunchbase funding < 2M, pdl founded after 2019, and pdl employee count < 20
# predict (ChatGPT) whether DevTool or DeepTech
# label companies that are neither as 'company was neither deeptech nor a devtool'
# make predictions (ChatGPT) for the companies that are DevTool and/or DeepTech
# make another column for filtered predictions (categories that were in the original list)

# the above pipeline will also work for single companies where data from crunchbase, pdl, etc. is included. It will simply
# run the pipeline on a one-row dataframe

## for a single company (just the website description):
# predict (ChatGPT) whether DevTool or DeepTech
# if neither, return 'company was neither deeptech nor a devtool' as prediction
# otherwise, make prediction (ChatGPT)
# also return the filtered predictions (categories that were in the original list)

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access variables using os.environ
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

category_list_df = pd.read_csv('full_categories_with_embeddings.csv')

category_list = category_list_df['category'].tolist()
# print("Category list: ", category_list)

sample_companies_dataframe = pd.read_csv('sample_companies_dataframe.csv')

class YesNo(BaseModel):
    answer: str = Field(description="Answer 'Yes' if your answer is affirmative, and 'No' if it is not")

class Output(BaseModel):
    most_related_tags: list = Field(description="the tags most related to the company (just the tag without explanation)")
    least_related_tags: list = Field(description="the tags least related to the company (just the tag without explanation)")

def is_devtool(query):
    # formatted_tags = ', '.join(category_list)
    model_name = 'gpt-4'
    temperature = 0
    model = OpenAI(model_name=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY)
    parser = PydanticOutputParser(pydantic_object=YesNo)
    prompt = PromptTemplate(
        template="I am working in a venture capital firm. Our fund is focused on investing in developer tools. Below I will give you a summary of the website of a company. Please tell me whether this company is in our fund's focus. Start with a yes/no answer, and then explain. \n{format_instructions}\n Here is the website summary: {query}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    _input = prompt.format_prompt(query=query)
    # print("INPUT", _input)
    response = model(_input.to_string())
    # print("RESPONSE", response)
    response = parser.parse(response).answer
    return response

def is_deeptech(query):
    # formatted_tags = ', '.join(category_list)
    model_name = 'gpt-4'
    temperature = 0
    model = OpenAI(model_name=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY)
    parser = PydanticOutputParser(pydantic_object=YesNo)
    prompt = PromptTemplate(
        template="I am working in a venture capital firm. Our fund is focused on investing in deep tech startups: investing in novel software, science based startups, startups that have a large R&D component. Below I will give you a summary of the website of a company. Please tell me whether this company is in our fund's focus. Start with a yes/no answer, and then explain. \n{format_instructions}\n Here is the website summary: {query}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    _input = prompt.format_prompt(query=query)
    # print("INPUT", _input)
    response = model(_input.to_string())
    # print("RESPONSE", response)
    response = parser.parse(response).answer
    return response

def filter_dataframe(companies_dataframe):
    crunchbase_funding_condition = (companies_dataframe['total_funding_crunchbase'] < 2e6) | (companies_dataframe['total_funding_crunchbase'].isna())
    founded_condition = (companies_dataframe['pdl_founded'] > 2019) | (companies_dataframe['pdl_founded'].isna())
    employee_count_condition = (companies_dataframe['employee_count_pdl'] < 20) | (companies_dataframe['employee_count_pdl'].isna())
    filtered_dataframe = companies_dataframe[crunchbase_funding_condition & founded_condition & employee_count_condition]
    filtered_dataframe = filtered_dataframe.reset_index(drop=True)
    return filtered_dataframe

def make_binary_predictions(outfile, companies_dataframe):
    # make sure to specify that the outfile is a csv, i.e. 'outfile.csv', NOT 'outfile'
    companies_dataframe['Deeptech Prediction'] = companies_dataframe['Website Summary'].apply(is_deeptech)
    companies_dataframe['DevTool Prediction'] = companies_dataframe['Website Summary'].apply(is_devtool)
    companies_dataframe.to_csv(outfile, index=False)

def predict_category(query):
    formatted_tags = ', '.join(category_list)
    model_name = 'gpt-4'
    temperature = 0
    model = OpenAI(model_name=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY)
    parser = PydanticOutputParser(pydantic_object=Output)
    prompt = PromptTemplate(
        template="I will give you a company description. I will also provide you with a list of tags. Please select 2-7 of these tags that are most relevant to the company, and 2-7 tags that are least relevant to the company. Please do not feel the need to always find 7 tags- select as many or as few as you think are relevant. Please explain your reasoning for choosing each tag. Here is the list of tags: " + formatted_tags + " \n{format_instructions}\n Here is the company description and keywords list: {query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    _input = prompt.format_prompt(query=query)
    # print("INPUT", _input)
    response = model(_input.to_string())
    # print("RESPONSE", response)
    try:
        response = parser.parse(response).most_related_tags
        # print("RESPONSE", response)
        # print("TYPE", type(response))
    except:
        response = 'Failed to parse response'
    return response

def dataframe_pipeline(outfile, companies_dataframe):
    companies_dataframe_copy = companies_dataframe.copy()
    filtered_dataframe = filter_dataframe(companies_dataframe_copy)
    make_binary_predictions('binary_predictions.csv', filtered_dataframe)
    filtered_dataframe['Predictions'] = filtered_dataframe.apply(lambda row: 
    ['Company is neither deeptech nor a devtool'] if 'No' in row['Deeptech Prediction'] and 'No' in row['DevTool Prediction']
    else predict_category(row['Website Summary']), axis=1)
    filtered_dataframe['Predictions from Original List'] = filtered_dataframe.apply(lambda row: 
    ['Company is neither deeptech nor a devtool'] if 'No' in row['Deeptech Prediction'] and 'No' in row['DevTool Prediction']
    else [prediction for prediction in row['Predictions'] if prediction in category_list], axis=1)
    filtered_dataframe.to_csv(outfile, index=False)
    return filtered_dataframe

def single_company_pipeline(company_description):
    deeptech_prediction = is_deeptech(company_description)
    devtool_prediction = is_devtool(company_description)
    if 'No' in deeptech_prediction and 'No' in devtool_prediction:
        return 'Company is neither deeptech nor a devtool'
    else:
        predictions = predict_category(company_description)
        predictions_from_original_list = [prediction for prediction in predictions if prediction in category_list]
        ### need to filter by whether they are in original list
        return predictions, predictions_from_original_list

def predict_dataframe_selected_categories(categories_df, companies_df):
    ## assumes categories_df has a column 'category'
    category_list = categories_df['category'].tolist()
    companies_dataframe_copy = companies_df.copy()
    filtered_dataframe = filter_dataframe(companies_dataframe_copy)
    make_binary_predictions('binary_predictions.csv', filtered_dataframe)
    filtered_dataframe['Predictions'] = filtered_dataframe.apply(lambda row: 
    ['Company is neither deeptech nor a devtool'] if 'No' in row['Deeptech Prediction'] and 'No' in row['DevTool Prediction']
    else predict_category(row['Website Summary']), axis=1)
    filtered_dataframe['Predictions from Original List'] = filtered_dataframe.apply(lambda row: 
    ['Company is neither deeptech nor a devtool'] if 'No' in row['Deeptech Prediction'] and 'No' in row['DevTool Prediction']
    else [prediction for prediction in row['Predictions'] if prediction in category_list], axis=1)
    return filtered_dataframe


### RUN_MODE ###
# RUN_MODE 1 will run the dataframe pipeline on the selected number of companies from sample_companies_dataframe
# RUN_MODE 2 will run the single company pipeline on the specified company description
# RUN_MODE 3 will run the dataframe pipeline with the specified categories dataframe. Does NOT save as csv.

RUN_MODE = 3

if RUN_MODE == 1:
    test_companies = sample_companies_dataframe.head(2)
    predictions_df = dataframe_pipeline('dataframe_pipeline_test.csv', test_companies)
    print(predictions_df)
elif RUN_MODE == 2:
    company_description = 'Orbital Materials is a company that develops and commercialises innovative technologies to keep Earth our home forever. Our solutions focus on clean air, water and energy with the help of AI and generative language models. We are committed to providing sustainable fuels, carbon capture and removal of harmful chemicals from the environment. Our team of experts is devoted to creating novel solutions to ensure Earth remains our home for the future.'
    predictions, predictions_from_original_list = single_company_pipeline(company_description)
    print("Predictions: ", predictions)
    print("Predictions from original list: ", predictions_from_original_list)
elif RUN_MODE == 3:
    test_companies = sample_companies_dataframe.head(2)
    predictions_df = predict_dataframe_selected_categories(category_list_df, test_companies)
    print(predictions_df)