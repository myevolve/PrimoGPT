import os
from typing import List, Dict
import json
import random
import csv
import re
from tqdm import tqdm

import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
from unsloth import FastLanguageModel

import json

from primogpt.prepare_data import get_company_profile

load_dotenv()
openai_key = os.environ.get("OPENAI_KEY")

def map_bin_label(bin_lb):    
    # Replacing 'U' and 'D' labels with descriptions 'up by ' and 'down by ', respectively
    lb = bin_lb.replace('U', 'up by ')
    lb = lb.replace('D', 'down by ')
    
    # Replacing numbers with percentage ranges
    lb = lb.replace('1', '0-1%')
    lb = lb.replace('2', '1-2%')
    lb = lb.replace('3', '2-3%')
    lb = lb.replace('4', '3-4%')
    
    # Special handling for changes exceeding 5%, marked with '5+'
    if lb.endswith('+'):
        lb = lb.replace('5+', 'more than 5%')
    else:
        # For cases that exactly match '5', marking the range as '4-5%'
        lb = lb.replace('5', '4-5%')
    
    # Returns the descriptive change label
    return lb

def format_prediction_string(row: pd.Series, next_row: pd.Series) -> str:
    # Mapping bin label to descriptive format
    bin_description = map_bin_label(next_row['Bin Label'])
    
    # Formatting dates
    current_date = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
    next_date = pd.to_datetime(next_row['Date']).strftime('%Y-%m-%d')
    
    # Creating the final string
    formatted_string = f"""
    From today's trading day ({current_date}) to the next trading day ({next_date}), the stock price will be {bin_description} to ${next_row['Adj Close Price']:.2f} from ${row['Adj Close Price']:.2f}. 
    Keep this information in mind when generating features, and try to identify patterns or factors in the news and press releases that might explain or correlate with this price movement.
    Use this future price information as context, but base your feature generation primarily on the content and implications of the news and press releases.
    """
    
    return formatted_string

def format_company_info(profile: Dict, adj_close: float, bin_label: str) -> str:
    price_change_description = map_bin_label(bin_label)
    return f"""{profile['name']} is a company trading under the ticker {profile['symbol']}.
    The company operates in the {profile['industry']} industry with a market capitalization of ${profile['marketCapitalization']:,.0f}.
    It has {profile['employeeTotal']:,} employees.
    The current stock price is ${adj_close:.2f}, with today's price change {price_change_description} compared to the previous closing price."""

def get_news_for_date(row: Dict) -> List[Dict]:
    date = row['Date'] if isinstance(row['Date'], str) else row['Date'].strftime('%Y-%m-%d')
    news = json.loads(row["News"])
    return [
        {"headline": n['headline'], "summary": n['summary'], "date": n['date']}
        for n in news
        if n['date'][:8] <= date.replace('-', '') and
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")
    ]

def sample_news(news: List[Dict], k: int = 5) -> List[Dict]:
    return random.sample(news, min(k, len(news)))

def format_news(news: List[Dict]) -> str:
    return "\n".join([f"[Headline]: {n['headline']}\n[Summary]: {n['summary']}" for n in news])

def get_press_releases_for_date(row: Dict) -> List[Dict]:
    date = row['Date'] if isinstance(row['Date'], str) else row['Date'].strftime('%Y-%m-%d')
    press_releases = json.loads(row["PressReleases"])
    return [
        {"date": pr['date'], "headline": pr['headline'], "description": pr['description']}
        for pr in press_releases
        if pr['date'][:8] <= date.replace('-', '')
    ]

def format_press_releases(press_releases: List[Dict]) -> str:
    return "\n".join([f"[Headline]: {pr['headline']}\n[Description]: {pr['description']}" for pr in press_releases])

def prepare_input(row: Dict, next_row: Dict, profile: Dict) -> Dict:
    news = get_news_for_date(row)
    sampled_news = sample_news(news, k=10)
    press_releases = get_press_releases_for_date(row)
    
    prediction_string = format_prediction_string(pd.Series(row), pd.Series(next_row))
    
    return {
        "company_info": format_company_info(profile, row['Adj Close Price'], row['Bin Label']),
        "trading_day": row['Date'],
        "symbol": profile['symbol'],
        "adj_close": row['Adj Close Price'],
        "news": format_news(sampled_news),
        "press_releases": format_press_releases(press_releases),
        "prediction_string": prediction_string
    }

def initialize_csv(csv_file: str):
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Date", "Adj Close Price", "Returns", "Bin Label", 
                "News Relevance", "Sentiment", "Price Impact Potential", 
                "Trend Direction", "Earnings Impact", "Investor Confidence", "Risk Profile Change",
                "Prompt"
            ])

def save_results_to_csv(csv_file: str, results: List[Dict]):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for result in results:
            writer.writerow([
                result["Date"],
                result["Adj Close Price"],
                result["Returns"],
                result["Bin Label"],
                result["News Relevance"],
                result["Sentiment"],
                result["Price Impact Potential"],
                result["Trend Direction"],
                result["Earnings Impact"],
                result["Investor Confidence"],
                result["Risk Profile Change"],
                result["Prompt"]
            ])

def create_json_input_template():
    return """
            [COMPANY BASICS]
            {company_info}

            [RECENT NEWS]
            Here are the recent news articles related to {symbol}:
            {news}

            [LATEST PRESS RELEASE]
            Here is the most recent press release from {symbol} (if available):
            {press_releases}

            [ANALYSIS TASKS]
            Based primarily on the provided news and press releases (if available) generate the following features defined in the output format. When analyzing, pay special attention to:

            1. Potential negative impacts or risks, even if they're subtle or not the main focus of the news.
            2. Market saturation signs, increased competition, or regulatory challenges.
            3. Discrepancies between the tone of the news and the actual content.
            4. Possible overoptimism in positive news that might lead to unrealistic expectations.
            5. Short-term versus long-term implications of the news, especially potential short-term negative reactions.

            Be cautious of overly positive sentiment and ensure you're giving appropriate weight to any negative indicators.

            [FUTURE INFORMATION]
            {prediction_string}

            [OUTPUT FORMAT]
            The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ""```json"" and ""```"":

            ```json
            {{
                ""news_relevance"": string  // How directly relevant the provided news and press releases are to the stock's performance (0: not relevant, 1: somewhat relevant, 2: highly relevant)
                ""sentiment"": string  // Overall sentiment towards the stock based on the news and press releases (-1: negative, 0: neutral, 1: positive)
                ""price_impact_potential"": string  // Potential impact of the news and press releases on the stock's price in the next trading session (-3: strong negative, -2: moderate negative, -1: slight negative, 0: no impact, 1: slight positive, 2: moderate positive, 3: strong positive)
                ""trend_direction"": string  // Likely direction of the stock's price trend based on the news and press releases (-1: downward, 0: neutral, 1: upward)
                ""earnings_impact"": string  // Potential impact of the news on the company's future earnings (-2: significant negative, -1: slight negative, 0: neutral or unclear, 1: slight positive, 2: significant positive)
                ""investor_confidence"": string  // How the news might affect investor confidence in the company (-3: major decrease, -2: moderate decrease, -1: slight decrease, 0: no change, 1: slight increase, 2: moderate increase, 3: major increase)
                ""risk_profile_change"": string  // How the news might change the perceived risk profile of the company (-2: significantly increased risk, -1: slightly increased risk, 0: no significant change, 1: slightly decreased risk, 2: significantly decreased risk)
            }}
            ```

            IMPORTANT: Your response should ONLY include the JSON structure as specified above. Do not include any additional explanation or analysis.
            """

def clean_input_string(input_str):
    # Removing multiple spaces and newlines
    cleaned = re.sub(r'\s+', ' ', input_str)
    # Removing spaces at the beginning and end
    cleaned = cleaned.strip()
    return cleaned

def save_results_to_json(json_file: str, results: List[Dict]):
    formatted_results = []
    for result in results:
        response_json = {
            "news_relevance": result['News Relevance'],
            "sentiment": result['Sentiment'],
            "price_impact_potential": result['Price Impact Potential'],
            "trend_direction": result['Trend Direction'],
            "earnings_impact": result['Earnings Impact'],
            "investor_confidence": result['Investor Confidence'],
            "risk_profile_change": result['Risk Profile Change']
        }
        response_string = json.dumps(response_json, ensure_ascii=False)

        instruction_string = f"""
        [SYSTEM PROMPT]
        You are a senior quantitative analyst specializing in stock market analysis.
        Your task is to analyze the provided company recent news and press releases to generate key features that could influence the stock's price movement in the next trading session.
        Focus on interpreting the given data to provide insights for algorithmic trading models.
        """

        formatted_result = {
            "instruction": clean_input_string(instruction_string),
            "input": clean_input_string(result["JSON Input"]),
            "response": response_string
        }
        formatted_results.append(formatted_result)
    
    with open(json_file, 'w') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)

def create_prompt_template():
    return ChatPromptTemplate.from_template("""
    [SYSTEM PROMPT]
    You are a senior quantitative analyst specializing in stock market analysis.
    Your task is to analyze the provided company recent news and press releases to generate key features that could influence the stock's price movement in the next trading session.
    Focus on interpreting the given data to provide insights for algorithmic trading models.

    [COMPANY BASICS]
    {company_info}

    [RECENT NEWS]
    Here are the recent news articles related to {symbol}:
    {news}

    [LATEST PRESS RELEASE]
    Here is the most recent press release from {symbol}:
    {press_releases}
                                            
    [ANALYSIS TASKS]
    Based primarily on the provided news and press releases (if available) for {symbol} generate the following features defined in the output format. When analyzing, pay special attention to:

    1. Potential negative impacts or risks, even if they're subtle or not the main focus of the news.
    2. Market saturation signs, increased competition, or regulatory challenges.
    3. Discrepancies between the tone of the news and the actual content.
    4. Possible overoptimism in positive news that might lead to unrealistic expectations.
    5. Short-term versus long-term implications of the news, especially potential short-term negative reactions.

    Be cautious of overly positive sentiment and ensure you're giving appropriate weight to any negative indicators.
                                            
    [OUTPUT FORMAT]
    {format_instructions}

    IMPORTANT: Your response should ONLY include the JSON structure as specified above. Do not include any additional explanation or analysis.
    """)

def create_prompt_template_for_train():
    return ChatPromptTemplate.from_template("""
    [SYSTEM PROMPT]
    You are a senior quantitative analyst specializing in stock market analysis.
    Your task is to analyze the provided company recent news and press releases to generate key features that could influence the stock's price movement in the next trading session.
    Focus on interpreting the given data to provide insights for algorithmic trading models.

    [COMPANY BASICS]
    {company_info}

    [RECENT NEWS]
    Here are the recent news articles related to {symbol}:
    {news}

    [LATEST PRESS RELEASE]
    Here is the most recent press release from {symbol} (if available):
    {press_releases}
                                            
    [ANALYSIS TASKS]
    Based primarily on the provided news and press releases (if available) for {symbol} generate the following features defined in the output format. When analyzing, pay special attention to:

    1. Potential negative impacts or risks, even if they're subtle or not the main focus of the news.
    2. Market saturation signs, increased competition, or regulatory challenges.
    3. Discrepancies between the tone of the news and the actual content.
    4. Possible overoptimism in positive news that might lead to unrealistic expectations.
    5. Short-term versus long-term implications of the news, especially potential short-term negative reactions.

    Be cautious of overly positive sentiment and ensure you're giving appropriate weight to any negative indicators.  
                                                                               
    [FUTURE INFORMATION]
    {prediction_string}

    [OUTPUT FORMAT]
    {format_instructions}

    IMPORTANT: Your response should ONLY include the JSON structure as specified above. Do not include any additional explanation or analysis.
    """)

def create_output_parser():
    response_schemas = [
        ResponseSchema(name="news_relevance", description="How directly relevant the provided news and press releases are to the stock's performance (0: not relevant, 1: somewhat relevant, 2: highly relevant)"),
        ResponseSchema(name="sentiment", description="Overall sentiment towards the stock based on the news and press releases (-1: negative, 0: neutral, 1: positive)"),
        ResponseSchema(name="price_impact_potential", description="Potential impact of the news and press releases on the stock's price in the next trading session (-3: strong negative, -2: moderate negative, -1: slight negative, 0: no impact, 1: slight positive, 2: moderate positive, 3: strong positive)"),
        ResponseSchema(name="trend_direction", description="Likely direction of the stock's price trend based on the news and press releases (-1: downward, 0: neutral, 1: upward)"),
        ResponseSchema(name="earnings_impact", description="Potential impact of the news on the company's future earnings (-2: significant negative, -1: slight negative, 0: neutral or unclear, 1: slight positive, 2: significant positive)"),
        ResponseSchema(name="investor_confidence", description="How the news might affect investor confidence in the company (-3: major decrease, -2: moderate decrease, -1: slight decrease, 0: no change, 1: slight increase, 2: moderate increase, 3: major increase)"),
        ResponseSchema(name="risk_profile_change", description="How the news might change the perceived risk profile of the company (-2: significantly increased risk, -1: slightly increased risk, 0: no significant change, 1: slightly decreased risk, 2: significantly decreased risk)")
    ]
    return StructuredOutputParser.from_response_schemas(response_schemas)

def process_stock_data(symbol: str, data_dir: str, start_date: str, end_date: str, is_for_train: bool = False, custom_gpt: bool = False):
    csv_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_gpt.csv'
    json_file = f'{data_dir}/{symbol}_{start_date}_{end_date}_gpt.json'
    
    # Initialize CSV file
    initialize_csv(csv_file)
    
    df = pd.read_csv(f'{data_dir}/{symbol}_{start_date}_{end_date}.csv')
    profile = get_company_profile(symbol)

    output_parser = create_output_parser()

    if is_for_train:
        prompt = create_prompt_template_for_train()
    else:
        prompt = create_prompt_template()

    json_input_template = create_json_input_template()

    if custom_gpt:
        # Load custom GPT model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="One2Many/PrimoGPT-Instruct",
            max_seq_length=8192,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

        # Defining Alpaca prompt format
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        # Define prompt template for custom GPT
        custom_instruction_template = """
                [SYSTEM PROMPT]
                You are a senior quantitative analyst specializing in stock market analysis.
                Your task is to analyze the provided company recent news and press releases to generate key features that could influence the stock's price movement in the next trading session.
                Focus on interpreting the given data to provide insights for algorithmic trading models.
                """

        custom_input_template = """
                [COMPANY BASICS]
                {company_info}

                [RECENT NEWS]
                Here are the recent news articles related to {symbol}:
                {news}

                [LATEST PRESS RELEASE]
                Here is the most recent press release from {symbol}:
                {press_releases}

                [ANALYSIS TASKS]
                Based primarily on the provided news and press releases (if available) generate the following features defined in the output format. When analyzing, pay special attention to:

                1. Potential negative impacts or risks, even if they're subtle or not the main focus of the news.
                2. Market saturation signs, increased competition, or regulatory challenges.
                3. Discrepancies between the tone of the news and the actual content.
                4. Possible overoptimism in positive news that might lead to unrealistic expectations.
                5. Short-term versus long-term implications of the news, especially potential short-term negative reactions.

                Be cautious of overly positive sentiment and ensure you're giving appropriate weight to any negative indicators.

                [OUTPUT FORMAT]
                The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ""```json"" and ""```"":

                ```json
                {{
                    ""news_relevance"": string  // How directly relevant the provided news and press releases are to the stock's performance (0: not relevant, 1: somewhat relevant, 2: highly relevant)
                    ""sentiment"": string  // Overall sentiment towards the stock based on the news and press releases (-1: negative, 0: neutral, 1: positive)
                    ""price_impact_potential"": string  // Potential impact of the news and press releases on the stock's price in the next trading session (-3: strong negative, -2: moderate negative, -1: slight negative, 0: no impact, 1: slight positive, 2: moderate positive, 3: strong positive)
                    ""trend_direction"": string  // Likely direction of the stock's price trend based on the news and press releases (-1: downward, 0: neutral, 1: upward)
                    ""earnings_impact"": string  // Potential impact of the news on the company's future earnings (-2: significant negative, -1: slight negative, 0: neutral or unclear, 1: slight positive, 2: significant positive)
                    ""investor_confidence"": string  // How the news might affect investor confidence in the company (-3: major decrease, -2: moderate decrease, -1: slight decrease, 0: no change, 1: slight increase, 2: moderate increase, 3: major increase)
                    ""risk_profile_change"": string  // How the news might change the perceived risk profile of the company (-2: significantly increased risk, -1: slightly increased risk, 0: no significant change, 1: slightly decreased risk, 2: significantly decreased risk)
                }}
                ```

                IMPORTANT: Your response should ONLY include the JSON structure as specified above. Do not include any additional explanation or analysis.
                """

        results = []
        for i in tqdm(range(len(df) - 1), total=len(df) - 1):
            row = df.iloc[i].to_dict()
            next_row = df.iloc[i + 1].to_dict()
            
            input_data = prepare_input(row, next_row, profile)
            formatted_prompt = custom_input_template.format(
                company_info=input_data['company_info'],
                symbol=input_data['symbol'],
                news=input_data['news'],
                press_releases=input_data['press_releases']
            )

            max_retries = 6  # Set the maximum number of retry attempts
            for attempt in range(max_retries):
                inputs = tokenizer([
                    alpaca_prompt.format(
                        custom_instruction_template,
                        formatted_prompt,
                        "",
                    )
                ], return_tensors = "pt").to("cuda")
                
                outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                try:
                    response_match = re.search(r'### Response:(.*)', response, re.DOTALL)
                    if response_match:
                        response_content = response_match.group(1).strip()
                        
                        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            output = json.loads(json_str)
                            
                            results.append({
                                "Date": row['Date'],
                                "Adj Close Price": row['Adj Close Price'],
                                "Returns": row['Returns'],
                                "Bin Label": row['Bin Label'],
                                "News Relevance": int(output.get("news_relevance", 0)),
                                "Sentiment": int(output.get("sentiment", 0)),
                                "Price Impact Potential": int(output.get("price_impact_potential", 0)),
                                "Trend Direction": int(output.get("trend_direction", 0)),
                                "Earnings Impact": int(output.get("earnings_impact", 0)),
                                "Investor Confidence": int(output.get("investor_confidence", 0)),
                                "Risk Profile Change": int(output.get("risk_profile_change", 0)),
                                "Prompt": formatted_prompt,
                                "JSON Input": json.dumps(input_data)
                            })
                            break  # Successfully processed, exit retry loop
                        else:
                            print(f"No JSON found in the response for row {i}, attempt {attempt + 1}")
                    else:
                        print(f"No response found in the output for row {i}, attempt {attempt + 1}")
                except Exception as e:
                    print(f"Error processing row {i}, attempt {attempt + 1}: {e}")
                
                if attempt == max_retries - 1:
                    print(f"Failed to process row {i} after {max_retries} attempts")
                    results.append({
                        "Date": row['Date'],
                        "Adj Close Price": row['Adj Close Price'],
                        "Returns": row['Returns'],
                        "Bin Label": row['Bin Label'],
                        "News Relevance": 0,
                        "Sentiment": 0,
                        "Price Impact Potential": 0,
                        "Trend Direction": 0,
                        "Earnings Impact": 0,
                        "Investor Confidence": 0,
                        "Risk Profile Change": 0,
                        "Prompt": formatted_prompt,
                        "JSON Input": json.dumps(input_data)
                    })

    else:
        model = ChatOpenAI(model="gpt-4o", api_key=openai_key)

        chain = (
            RunnablePassthrough.assign(format_instructions=output_parser.get_format_instructions)
            | prompt
            | model
            | output_parser
        )

        results = []
        for i in tqdm(range(len(df) - 1), total=len(df) - 1):
            row = df.iloc[i].to_dict()
            next_row = df.iloc[i + 1].to_dict()
            
            input_data = prepare_input(row, next_row, profile)
            try:
                output = chain.invoke(input_data)
            except Exception as e:
                print(f"Error processing row: {e}")
                output = None

            results.append({
                "Date": row['Date'],
                "Adj Close Price": row['Adj Close Price'],
                "Returns": row['Returns'],
                "Bin Label": row['Bin Label'],
                "News Relevance": output.get("news_relevance") if output else None,
                "Sentiment": output.get("sentiment") if output else None,
                "Price Impact Potential": output.get("price_impact_potential") if output else None,
                "Trend Direction": output.get("trend_direction") if output else None,
                "Earnings Impact": output.get("earnings_impact") if output else None,
                "Investor Confidence": output.get("investor_confidence") if output else None,
                "Risk Profile Change": output.get("risk_profile_change") if output else None,
                "Prompt": prompt.format(**input_data, format_instructions=output_parser.get_format_instructions()),
                "JSON Input": json_input_template.format(**input_data)
            })

    save_results_to_csv(csv_file, results)

    if is_for_train:
        save_results_to_json(json_file, results)

    return results