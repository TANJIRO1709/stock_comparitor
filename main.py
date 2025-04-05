from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from yahooquery import Ticker
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import re

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI and LLM
app = FastAPI()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Input model
class StockComparisonRequest(BaseModel):
    stock1: str
    stock2: str


def compare_stock(metrics: dict) -> dict:
    stock1, stock2 = list(metrics.keys())
    metric_names = list(metrics[stock1].keys())

    # Format metrics for prompt
    metric_lines = []
    for metric in metric_names:
        val1 = metrics[stock1].get(metric, "N/A")
        val2 = metrics[stock2].get(metric, "N/A")
        metric_lines.append(f"- {metric}: {stock1} = {val1}, {stock2} = {val2}")
    metric_text = "\n".join(metric_lines)

    prompt = f"""
    You are a financial analyst specializing in comparing stocks.
    Based on the following financial metrics, compare the two stocks: {stock1} and {stock2}.
    Your response should help a non-technical person understand which stock is better and why. Assume the user doesn't know anything about the metrics.

    Metrics:
    {metric_text}

    Respond in JSON format with:
    - better_stock: The stock symbol that is better based on the analysis
    - reasoning: A clear, simple explanation for your decision
    """

    response = llm.predict(prompt)

    # Try to extract valid JSON from response
    match = re.search(r'{\s*"better_stock":\s*".+?",\s*"reasoning":\s*".+?"\s*}', response, re.DOTALL)
    if match:
        extracted_json = match.group()
        return json.loads(extracted_json)
    else:
        return {
            "better_stock": None,
            "reasoning": "Could not extract JSON from LLM response.",
            "raw_response": response
        }


def get_stock_metrics(stock1: str, stock2: str) -> dict:
    ticker1 = Ticker(stock1)
    ticker2 = Ticker(stock2)

    metrics = ['forwardPE', 'trailingPE', 'dividendYield', 'beta', 'marketCap']

    summary1 = ticker1.summary_detail.get(stock1, {})
    summary2 = ticker2.summary_detail.get(stock2, {})

    if not summary1 or not summary2:
        raise ValueError("One or both stock symbols are invalid or missing data.")

    return {
        stock1: {metric: summary1.get(metric, 'N/A') for metric in metrics},
        stock2: {metric: summary2.get(metric, 'N/A') for metric in metrics}
    }


@app.post("/compare-stocks")
def compare_stocks_api(request: StockComparisonRequest):
    try:
        # Step 1: Get numeric metrics
        metrics = get_stock_metrics(request.stock1, request.stock2)

        # Step 2: Use LLM to get the better stock
        comparison = compare_stock(metrics)

        # Step 3: Return everything
        return {
            "stock1": request.stock1,
            "stock2": request.stock2,
            "metrics": metrics,
            "better_stock": comparison.get("better_stock"),
            "reasoning": comparison.get("reasoning")
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")